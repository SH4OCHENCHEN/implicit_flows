import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, Value


# ============================================================
# Core: Compute Drift V and Loss
# ============================================================

def compute_drift(
    gen: jnp.ndarray,
    pos: jnp.ndarray,
    neg: jnp.ndarray,
    temp: float = 0.05,
    exclude_self_neg: bool = False,
):
    """Compute drift field V with explicit positive and negative sets.

    Args:
        gen: Generated samples [..., G, D]
        pos: Positive samples [..., P, D]
        neg: Negative samples [..., N, D]
        temp: Temperature for softmax kernel.
        exclude_self_neg: Whether to mask diagonal in gen-neg distances.

    Returns:
        V: Drift vectors [..., G, D]
    """
    dist_pos = jnp.linalg.norm(gen[..., :, None, :] - pos[..., None, :, :], axis=-1)  # [..., G, P]
    dist_neg = jnp.linalg.norm(gen[..., :, None, :] - neg[..., None, :, :], axis=-1)  # [..., G, N]

    if exclude_self_neg:
        g = gen.shape[-2]
        n = neg.shape[-2]
        dist_neg = dist_neg + jnp.eye(g, n, dtype=dist_neg.dtype) * 1e6

    logits_pos = -dist_pos / temp
    logits_neg = -dist_neg / temp
    logits = jnp.concatenate([logits_pos, logits_neg], axis=-1)  # [..., G, P+N]

    a_row = jax.nn.softmax(logits, axis=-1)
    a_col = jax.nn.softmax(logits, axis=-2)
    a = jnp.sqrt(jnp.clip(a_row * a_col, a_min=1e-12))

    p = pos.shape[-2]
    a_pos = a[..., :p]
    a_neg = a[..., p:]

    w_pos = a_pos * a_neg.sum(axis=-1, keepdims=True)
    w_neg = a_neg * a_pos.sum(axis=-1, keepdims=True)

    drift_pos = w_pos @ pos
    drift_neg = w_neg @ neg
    return drift_pos - drift_neg


def drifting_loss(
    gen: jnp.ndarray,
    pos: jnp.ndarray,
    neg: jnp.ndarray,
    temp: float = 0.05,
    exclude_self_neg: bool = False,
):
    """Drifting loss: MSE(gen, stopgrad(gen + V))."""
    v = compute_drift(gen, pos, neg, temp=temp, exclude_self_neg=exclude_self_neg)
    target = jax.lax.stop_gradient(gen + v)
    return jnp.mean((gen - target) ** 2)


def multi_temp_drifting_loss(
    gen: jnp.ndarray,
    pos: jnp.ndarray,
    neg: jnp.ndarray,
    temps,
    exclude_self_neg: bool = False,
):
    """Sum drifting losses computed with multiple temperatures."""
    loss = jnp.asarray(0.0, dtype=gen.dtype)
    for temp in temps:
        loss = loss + drifting_loss(
            gen,
            pos,
            neg,
            temp=float(temp),
            exclude_self_neg=exclude_self_neg,
        )
    return loss


class CDPV3Agent(flax.struct.PyTreeNode):
    """CDP v3 agent with explicit positive/negative drift sets.

    Diff vs `cdp_v2`:
    - Drift is rewritten to use explicit `(gen, pos, neg)` sets.
    - Policy drift positives are top-Q behavior-generated candidates.
    - Policy drift negatives are policy-generated + uniform samples.
    - Uses multi-temperature drift loss (`drift_temps`) summed over temps.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def critic_loss(self, batch, grad_params, rng):
        """Compute the TD critic loss (FQL-style, without CQL)."""
        rng, sample_rng = jax.random.split(rng)
        # Critic bootstrap uses the policy actor (not the behavior actor).
        next_actions = self.sample_actions(batch['next_observations'], seed=sample_rng)
        next_actions = jnp.clip(next_actions, -1, 1)

        next_qs = self.network.select('target_critic')(batch['next_observations'], actions=next_actions)
        if self.config['q_agg'] == 'min':
            next_q = next_qs.min(axis=0)
        else:
            next_q = next_qs.mean(axis=0)

        target_q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_q

        q_data = self.network.select('critic')(batch['observations'], actions=batch['actions'], params=grad_params)
        bellman_loss = jnp.square(q_data - target_q).mean()

        critic_loss = bellman_loss

        return critic_loss, {
            'critic_loss': critic_loss,
            'bellman_loss': bellman_loss,
            'q_data_mean': q_data.mean(),
            'q_data_max': q_data.max(),
            'q_data_min': q_data.min(),
        }

    def actor_loss(self, batch, grad_params, rng):
        """Train behavior/policy actors with drifting losses and different positive sets."""
        batch_size, action_dim = batch['actions'].shape
        behavior_num_neg = self.config['behavior_num_neg']
        policy_pos_candidates = self.config['policy_pos_candidates']
        policy_num_pos = self.config['policy_num_pos']
        policy_num_neg = self.config['policy_num_neg']
        policy_num_uniform_neg = self.config['policy_num_uniform_neg']
        drift_temps = self.config['drift_temps']

        rng, behavior_noise_rng, policy_x_noise_rng, pos_rng, uniform_neg_rng, mse_rng = jax.random.split(
            rng, 6
        )

        # =========================
        # 1) Behavior actor loss:
        # positives = dataset action only (one positive)
        # =========================
        behavior_noises = jax.random.normal(
            behavior_noise_rng, (batch_size, behavior_num_neg, action_dim)
        )
        behavior_obs_repeat = jnp.repeat(
            batch['observations'][:, None, ...], behavior_num_neg, axis=1
        )
        behavior_obs_flat = behavior_obs_repeat.reshape(
            (batch_size * behavior_num_neg, *batch['observations'].shape[1:])
        )
        behavior_noises_flat = behavior_noises.reshape((batch_size * behavior_num_neg, action_dim))

        raw_behavior_actions_flat = self.network.select('actor_behavior_onestep_flow')(
            behavior_obs_flat, behavior_noises_flat, params=grad_params
        )
        raw_behavior_actions = raw_behavior_actions_flat.reshape(
            (batch_size, behavior_num_neg, action_dim)
        )
        behavior_pos_actions = batch['actions'][:, None, :]
        behavior_drift_loss = multi_temp_drifting_loss(
            raw_behavior_actions,
            behavior_pos_actions,
            raw_behavior_actions,
            temps=drift_temps,
            exclude_self_neg=True,
        )

        # =========================
        # 2) Policy actor loss:
        # positives are top-Q behavior-generated candidates
        # negatives are 32 self-generated + 4 uniformly sampled actions
        # =========================
        policy_neg_obs_repeat = jnp.repeat(
            batch['observations'][:, None, ...], policy_num_neg, axis=1
        )
        policy_neg_obs_flat = policy_neg_obs_repeat.reshape(
            (batch_size * policy_num_neg, *batch['observations'].shape[1:])
        )
        policy_x_noises = jax.random.normal(policy_x_noise_rng, (batch_size, policy_num_neg, action_dim))
        policy_x_noises_flat = policy_x_noises.reshape((batch_size * policy_num_neg, action_dim))
        raw_policy_actions_flat = self.network.select('actor_onestep_flow')(
            policy_neg_obs_flat, policy_x_noises_flat, params=grad_params
        )
        raw_policy_actions = raw_policy_actions_flat.reshape((batch_size, policy_num_neg, action_dim))
        raw_policy_neg_actions = jnp.clip(raw_policy_actions, -1, 1)

        # Candidate positives are generated by behavior actor.
        pos_noises = jax.random.normal(pos_rng, (batch_size, policy_pos_candidates, action_dim))
        pos_obs_repeat = jnp.repeat(
            batch['observations'][:, None, ...], policy_pos_candidates, axis=1
        )
        pos_obs_flat = pos_obs_repeat.reshape(
            (batch_size * policy_pos_candidates, *batch['observations'].shape[1:])
        )
        pos_noises_flat = pos_noises.reshape((batch_size * policy_pos_candidates, action_dim))

        behavior_pos_actions_flat = self.network.select('actor_behavior_onestep_flow')(
            pos_obs_flat, pos_noises_flat, params=grad_params
        )
        behavior_pos_actions_flat = jnp.clip(behavior_pos_actions_flat, -1, 1)
        behavior_pos_actions = behavior_pos_actions_flat.reshape(
            (batch_size, policy_pos_candidates, action_dim)
        )

        behavior_pos_qs = self.network.select('critic')(pos_obs_flat, actions=behavior_pos_actions_flat)
        if self.config['q_agg'] == 'min':
            behavior_pos_q = behavior_pos_qs.min(axis=0)
        else:
            behavior_pos_q = behavior_pos_qs.mean(axis=0)
        behavior_pos_q = behavior_pos_q.reshape((batch_size, policy_pos_candidates))
        behavior_pos_q = jax.lax.stop_gradient(behavior_pos_q)

        # Keep top-k Q behavior actions as policy positives.
        pos_topk = min(policy_num_pos, policy_pos_candidates)
        top_pos_q, pos_idx = jax.lax.top_k(behavior_pos_q, pos_topk)
        pos_idx_expanded = jnp.expand_dims(pos_idx, axis=-1)
        pos_idx_expanded = jnp.repeat(pos_idx_expanded, action_dim, axis=-1)
        policy_pos_actions = jnp.take_along_axis(behavior_pos_actions, pos_idx_expanded, axis=1)

        # Policy negatives = self-generated + uniform actions.
        uniform_policy_neg_actions = jax.random.uniform(
            uniform_neg_rng,
            (batch_size, policy_num_uniform_neg, action_dim),
            minval=-1.0,
            maxval=1.0,
        )
        policy_neg_actions = jnp.concatenate(
            [raw_policy_neg_actions, uniform_policy_neg_actions], axis=1
        )

        policy_drift_loss = multi_temp_drifting_loss(
            raw_policy_actions,
            policy_pos_actions,
            policy_neg_actions,
            temps=drift_temps,
            exclude_self_neg=True,
        )

        # Total actor loss: train both actors together.
        actor_loss = (
            self.config['drift_batch_weight'] * behavior_drift_loss
            + self.config['drift_prob_weight'] * policy_drift_loss
        )

        # MSE is measured on policy actor outputs.
        actions = self.sample_actions(batch['observations'], seed=mse_rng)
        mse = jnp.mean((actions - batch['actions']) ** 2)

        return actor_loss, {
            'actor_loss': actor_loss,
            'behavior_drift_loss': behavior_drift_loss,
            'policy_drift_loss': policy_drift_loss,
            'drift_batch_loss': behavior_drift_loss,  # backward-compatible key
            'drift_prob_loss': policy_drift_loss,  # backward-compatible key
            'pos_q': behavior_pos_q.mean(),
            'pos_q_sampled': top_pos_q.mean(),
            'policy_neg_raw_count': jnp.asarray(policy_num_neg),
            'policy_neg_uniform_count': jnp.asarray(policy_num_uniform_neg),
            'policy_neg_behavior_count': jnp.asarray(0),  # backward-compatible key
            'mse': mse,
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = critic_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'critic')

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions from the policy one-step actor."""
        seed, action_seed = jax.random.split(seed)
        noises = jax.random.normal(
            action_seed,
            (
                *observations.shape[: -len(self.config['ob_dims'])],
                self.config['action_dim'],
            ),
        )
        actions = self.network.select('actor_onestep_flow')(observations, noises)
        actions = jnp.clip(actions, -1, 1)
        return actions

    @jax.jit
    def sample_behavior_actions(
        self,
        observations,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions from the behavior one-step actor."""
        seed, action_seed = jax.random.split(seed)
        noises = jax.random.normal(
            action_seed,
            (
                *observations.shape[: -len(self.config['ob_dims'])],
                self.config['action_dim'],
            ),
        )
        actions = self.network.select('actor_behavior_onestep_flow')(observations, noises)
        actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(
        cls,
        seed,
        example_batch,
        config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            example_batch: Example batch.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_observations = example_batch['observations']
        ex_actions = example_batch['actions']
        ob_dims = ex_observations.shape[1:]
        action_dim = ex_actions.shape[-1]

        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = encoder_module()
            encoders['actor_onestep_flow'] = encoder_module()
            encoders['actor_behavior_onestep_flow'] = encoder_module()

        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=2,
            encoder=encoders.get('critic'),
        )
        actor_onestep_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_onestep_flow'),
        )
        actor_behavior_onestep_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_behavior_onestep_flow'),
        )

        network_info = dict(
            critic=(critic_def, (ex_observations, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_actions)),
            actor_onestep_flow=(actor_onestep_flow_def, (ex_observations, ex_actions)),
            actor_behavior_onestep_flow=(actor_behavior_onestep_flow_def, (ex_observations, ex_actions)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_critic'] = params['modules_critic']

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='cdp_v3',  # Agent name.
            ob_dims=ml_collections.config_dict.placeholder(list),  # Observation dimensions (will be set automatically).
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (will be set automatically).
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            actor_layer_norm=True,  # Whether to use layer normalization for the actor.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            q_agg='min',  # Aggregation method for target Q values.\
            drift_temps=(0.1, 1.0, 10.0),  # Multi-temperature drift loss (summed over all temps).
            drift_batch_weight=1.0,  # Weight of behavior-actor drifting loss (single dataset positive).
            drift_prob_weight=1.0,  # Weight of policy-actor drifting loss (top-Q behavior positives).
            behavior_num_neg=16,  # Number of generated negatives for behavior drifting.
            policy_pos_candidates=64,  # Number of behavior-generated candidate positives.
            policy_num_pos=16,  # Number of top-Q behavior positives used by policy drifting.
            policy_num_neg=32,  # Number of policy self-generated negatives.
            policy_num_uniform_neg=4,  # Number of uniformly sampled negatives.
            behavior_drift_temp=50.0,  # Deprecated/unused in cdp_v3 actor loss (kept for CLI compatibility).
            actor_drift_temp=5.0,  # Deprecated/unused in cdp_v3 actor loss (kept for CLI compatibility).
            num_neg=16,  # Deprecated/unused in cdp_v3 actor loss (kept for CLI compatibility).
            num_samples=16,  # Deprecated/unused in cdp_v3 actor loss (kept for CLI compatibility).
            pos_topk=4,  # Deprecated/unused in cdp_v3 actor loss (kept for CLI compatibility).
            pos_prob_temp=0.001,  # Deprecated/unused in cdp_v3 actor loss (kept for CLI compatibility).
            policy_neg_raw_ratio=4.0,  # Deprecated/unused in cdp_v3 actor loss (kept for CLI compatibility).
            policy_neg_behavior_ratio=1.0,  # Deprecated/unused in cdp_v3 actor loss (kept for CLI compatibility).
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
        )
    )
    return config
