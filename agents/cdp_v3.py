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


class CDPV3Agent(flax.struct.PyTreeNode):
    """CDP v3 agent with explicit positive/negative drift sets."""

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
        num_neg = self.config['num_neg']
        num_pos_samples = self.config['num_samples']
        pos_draws = min(self.config['pos_topk'], num_pos_samples)

        raw_neg_ratio = float(self.config['policy_neg_raw_ratio'])
        behavior_neg_ratio = float(self.config['policy_neg_behavior_ratio'])
        if num_neg <= 1 or behavior_neg_ratio <= 0:
            num_policy_neg_behavior = 0
            num_policy_neg_raw = num_neg
        else:
            num_policy_neg_behavior = int(
                round(num_neg * behavior_neg_ratio / (raw_neg_ratio + behavior_neg_ratio))
            )
            num_policy_neg_behavior = max(1, min(num_neg - 1, num_policy_neg_behavior))
            num_policy_neg_raw = num_neg - num_policy_neg_behavior

        rng, behavior_noise_rng, policy_x_noise_rng, policy_neg_raw_noise_rng, pos_rng, sample_pos_rng, behavior_neg_rng, mse_rng = jax.random.split(rng, 8)

        # =========================
        # 1) Behavior actor loss:
        # positives = dataset action only (one positive)
        # =========================
        behavior_noises = jax.random.normal(behavior_noise_rng, (batch_size, num_neg, action_dim))
        obs_repeat = jnp.repeat(batch['observations'][:, None, ...], num_neg, axis=1)
        obs_flat = obs_repeat.reshape((batch_size * num_neg, *batch['observations'].shape[1:]))
        behavior_noises_flat = behavior_noises.reshape((batch_size * num_neg, action_dim))

        raw_behavior_actions_flat = self.network.select('actor_behavior_onestep_flow')(
            obs_flat, behavior_noises_flat, params=grad_params
        )
        raw_behavior_actions = raw_behavior_actions_flat.reshape((batch_size, num_neg, action_dim))
        behavior_pos_actions = batch['actions'][:, None, :]
        behavior_drift_loss = drifting_loss(
            raw_behavior_actions,
            behavior_pos_actions,
            raw_behavior_actions,
            temp=self.config['drift_temp'],
            exclude_self_neg=True,
        )

        # =========================
        # 2) Policy actor loss:
        # positives are sampled from behavior-generated candidates by exp(Q) probabilities
        # =========================
        policy_x_noises = jax.random.normal(policy_x_noise_rng, (batch_size, num_neg, action_dim))
        policy_x_noises_flat = policy_x_noises.reshape((batch_size * num_neg, action_dim))
        raw_policy_actions_flat = self.network.select('actor_onestep_flow')(
            obs_flat, policy_x_noises_flat, params=grad_params
        )
        raw_policy_actions = raw_policy_actions_flat.reshape((batch_size, num_neg, action_dim))

        # Candidate positives are generated by behavior actor.
        pos_noises = jax.random.normal(pos_rng, (batch_size, num_pos_samples, action_dim))
        pos_obs_repeat = jnp.repeat(batch['observations'][:, None, ...], num_pos_samples, axis=1)
        pos_obs_flat = pos_obs_repeat.reshape((batch_size * num_pos_samples, *batch['observations'].shape[1:]))
        pos_noises_flat = pos_noises.reshape((batch_size * num_pos_samples, action_dim))

        behavior_pos_actions_flat = self.network.select('actor_behavior_onestep_flow')(
            pos_obs_flat, pos_noises_flat, params=grad_params
        )
        behavior_pos_actions_flat = jnp.clip(behavior_pos_actions_flat, -1, 1)
        behavior_pos_actions = behavior_pos_actions_flat.reshape((batch_size, num_pos_samples, action_dim))

        behavior_pos_qs = self.network.select('critic')(pos_obs_flat, actions=behavior_pos_actions_flat)
        if self.config['q_agg'] == 'min':
            behavior_pos_q = behavior_pos_qs.min(axis=0)
        else:
            behavior_pos_q = behavior_pos_qs.mean(axis=0)
        behavior_pos_q = behavior_pos_q.reshape((batch_size, num_pos_samples))
        behavior_pos_q = jax.lax.stop_gradient(behavior_pos_q)

        # Build sampling distribution with state-wise scaled exp(Q).
        # Each state uses its own lambda over the num_pos_samples axis.
        lam = jax.lax.stop_gradient(
            1.0 / (jnp.abs(behavior_pos_q).mean(axis=1, keepdims=True) + 1e-8)
        )
        behavior_pos_q_scaled = behavior_pos_q * lam
        pos_probs = jax.nn.softmax(behavior_pos_q_scaled / self.config['pos_prob_temp'], axis=1)
        pos_probs = jax.lax.stop_gradient(pos_probs)

        sample_keys = jax.random.split(sample_pos_rng, batch_size)

        def _sample_indices(k, p):
            return jax.random.choice(k, a=p.shape[0], shape=(pos_draws,), replace=False, p=p)

        pos_idx = jax.vmap(_sample_indices)(sample_keys, pos_probs)  # [B, K]
        pos_idx_expanded = jnp.expand_dims(pos_idx, axis=-1)
        pos_idx_expanded = jnp.repeat(pos_idx_expanded, action_dim, axis=-1)
        policy_pos_actions = jnp.take_along_axis(behavior_pos_actions, pos_idx_expanded, axis=1)

        # Policy negatives mix raw-policy samples and behavior-generated samples.
        policy_neg_raw_noises = jax.random.normal(
            policy_neg_raw_noise_rng, (batch_size, num_policy_neg_raw, action_dim)
        )
        policy_neg_raw_obs_repeat = jnp.repeat(
            batch['observations'][:, None, ...], num_policy_neg_raw, axis=1
        )
        policy_neg_raw_obs_flat = policy_neg_raw_obs_repeat.reshape(
            (batch_size * num_policy_neg_raw, *batch['observations'].shape[1:])
        )
        policy_neg_raw_noises_flat = policy_neg_raw_noises.reshape(
            (batch_size * num_policy_neg_raw, action_dim)
        )
        raw_policy_neg_actions_flat = self.network.select('actor_onestep_flow')(
            policy_neg_raw_obs_flat, policy_neg_raw_noises_flat, params=grad_params
        )
        raw_policy_neg_actions_flat = jnp.clip(raw_policy_neg_actions_flat, -1, 1)
        raw_policy_neg_actions = raw_policy_neg_actions_flat.reshape(
            (batch_size, num_policy_neg_raw, action_dim)
        )

        if num_policy_neg_behavior > 0:
            behavior_neg_noises = jax.random.normal(
                behavior_neg_rng, (batch_size, num_policy_neg_behavior, action_dim)
            )
            behavior_neg_obs_repeat = jnp.repeat(
                batch['observations'][:, None, ...], num_policy_neg_behavior, axis=1
            )
            behavior_neg_obs_flat = behavior_neg_obs_repeat.reshape(
                (batch_size * num_policy_neg_behavior, *batch['observations'].shape[1:])
            )
            behavior_neg_noises_flat = behavior_neg_noises.reshape(
                (batch_size * num_policy_neg_behavior, action_dim)
            )
            behavior_neg_actions_flat = self.network.select('actor_behavior_onestep_flow')(
                behavior_neg_obs_flat, behavior_neg_noises_flat, params=grad_params
            )
            behavior_neg_actions_flat = jnp.clip(behavior_neg_actions_flat, -1, 1)
            behavior_neg_actions = behavior_neg_actions_flat.reshape(
                (batch_size, num_policy_neg_behavior, action_dim)
            )
            policy_neg_actions = jnp.concatenate(
                [raw_policy_neg_actions, behavior_neg_actions], axis=1
            )
        else:
            policy_neg_actions = raw_policy_neg_actions

        policy_drift_loss = drifting_loss(
            raw_policy_actions,
            policy_pos_actions,
            policy_neg_actions,
            temp=self.config['drift_temp'],
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
            'pos_q_sampled': jnp.take_along_axis(behavior_pos_q, pos_idx, axis=1).mean(),
            'pos_q_scaled': behavior_pos_q_scaled.mean(),
            'pos_q_scaled_sampled': jnp.take_along_axis(behavior_pos_q_scaled, pos_idx, axis=1).mean(),
            'lam_mean': lam.mean(),
            'lam_min': lam.min(),
            'lam_max': lam.max(),
            'policy_neg_raw_count': jnp.asarray(num_policy_neg_raw),
            'policy_neg_behavior_count': jnp.asarray(num_policy_neg_behavior),
            'pos_entropy': (-pos_probs * jnp.log(pos_probs + 1e-12)).sum(axis=1).mean(),
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
            drift_temp=0.01,  # Temperature used in drifting BC.
            drift_batch_weight=1.0,  # Weight of behavior-actor drifting loss (single dataset positive).
            drift_prob_weight=1.0,  # Weight of policy-actor drifting loss (sampled behavior positives).
            num_neg=16,  # Number of negative/generated samples per state (same for both actors).
            num_samples=16,  # Number of behavior-actor candidate positives for policy actor.
            pos_topk=4,  # Number of sampled positives drawn from exp(Q) distribution.
            pos_prob_temp=0.001,  # Temperature for exp(Q) sampling probabilities.
            policy_neg_raw_ratio=4.0,  # Policy-neg mix ratio numerator for raw policy actions.
            policy_neg_behavior_ratio=1.0,  # Policy-neg mix ratio numerator for behavior actions.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
        )
    )
    return config
