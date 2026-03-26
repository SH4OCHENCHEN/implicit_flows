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

def compute_drift(gen: jnp.ndarray, pos: jnp.ndarray, temp: float = 0.05):
    """Compute drift field V with attention-based kernel.

    Args:
        gen: Generated samples [..., G, D]
        pos: Data samples [..., P, D]
        temp: Temperature for softmax kernel.

    Returns:
        V: Drift vectors [..., G, D]
    """
    targets = jnp.concatenate([gen, pos], axis=-2)  # [..., G+P, D]
    g = gen.shape[-2]

    diff = gen[..., :, None, :] - targets[..., None, :, :]  # [..., G, G+P, D]
    dist = jnp.linalg.norm(diff, axis=-1)  # [..., G, G+P]

    large_eye = jnp.eye(g, dtype=dist.dtype) * 1e6
    dist = dist.at[..., :, :g].add(large_eye)

    # Dimension-adaptive temperature: scale by sqrt(action_dim).
    action_dim = gen.shape[-1]
    adaptive_temp = temp * jnp.sqrt(jnp.asarray(action_dim, dtype=dist.dtype))
    kernel = jnp.exp(-dist / adaptive_temp)

    normalizer = kernel.sum(axis=-1, keepdims=True) * kernel.sum(axis=-2, keepdims=True)
    normalizer = jnp.sqrt(jnp.clip(normalizer, a_min=1e-12))
    normalized_kernel = kernel / normalizer

    pos_coeff = normalized_kernel[..., :, g:] * normalized_kernel[..., :, :g].sum(axis=-1, keepdims=True)
    pos_v = pos_coeff @ targets[..., g:, :]
    neg_coeff = normalized_kernel[..., :, :g] * normalized_kernel[..., :, g:].sum(axis=-1, keepdims=True)
    neg_v = neg_coeff @ targets[..., :g, :]

    return pos_v - neg_v


def drifting_loss(gen: jnp.ndarray, pos: jnp.ndarray, temp: float = 0.05):
    """Drifting loss: MSE(gen, stopgrad(gen + V))."""
    v = compute_drift(gen, pos, temp=temp)
    target = jax.lax.stop_gradient(gen + v)
    return jnp.mean((gen - target) ** 2)


class CDPV2Agent(flax.struct.PyTreeNode):
    """CDP v2 agent."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def critic_loss(self, batch, grad_params, rng):
        """Compute the TD critic loss (FQL-style, without CQL)."""
        rng, sample_rng = jax.random.split(rng)
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
        """Compute weighted drifting losses from batch-only and probability positives."""
        batch_size, action_dim = batch['actions'].shape
        num_neg = self.config['num_neg']
        num_pos_samples = self.config['num_samples']
        pos_draws = min(self.config['pos_topk'], num_pos_samples + 1)
        rng, noise_rng, pos_rng, sample_pos_rng, mse_rng = jax.random.split(rng, 5)

        # [B, G, D]: generate multiple raw actor actions per state.
        noises = jax.random.normal(noise_rng, (batch_size, num_neg, action_dim))
        obs_repeat = jnp.repeat(batch['observations'][:, None, ...], num_neg, axis=1)
        obs_flat = obs_repeat.reshape((batch_size * num_neg, *batch['observations'].shape[1:]))
        noises_flat = noises.reshape((batch_size * num_neg, action_dim))

        raw_actions_flat = self.network.select('actor_onestep_flow')(
            obs_flat, noises_flat, params=grad_params
        )
        raw_actor_actions = raw_actions_flat.reshape((batch_size, num_neg, action_dim))

        # Build pool = [batch action + sampled actions].
        pos_noises = jax.random.normal(pos_rng, (batch_size, num_pos_samples, action_dim))
        pos_obs_repeat = jnp.repeat(batch['observations'][:, None, ...], num_pos_samples, axis=1)
        pos_obs_flat = pos_obs_repeat.reshape((batch_size * num_pos_samples, *batch['observations'].shape[1:]))
        pos_noises_flat = pos_noises.reshape((batch_size * num_pos_samples, action_dim))

        sampled_pos_actions_flat = self.network.select('actor_onestep_flow')(
            pos_obs_flat, pos_noises_flat, params=grad_params
        )
        sampled_pos_actions_flat = jnp.clip(sampled_pos_actions_flat, -1, 1)
        sampled_pos_actions = sampled_pos_actions_flat.reshape((batch_size, num_pos_samples, action_dim))

        sampled_pos_qs = self.network.select('critic')(pos_obs_flat, actions=sampled_pos_actions_flat)
        if self.config['q_agg'] == 'min':
            sampled_pos_q = sampled_pos_qs.min(axis=0)
        else:
            sampled_pos_q = sampled_pos_qs.mean(axis=0)
        sampled_pos_q = sampled_pos_q.reshape((batch_size, num_pos_samples))

        batch_action_qs = self.network.select('critic')(batch['observations'], actions=batch['actions'])
        if self.config['q_agg'] == 'min':
            batch_action_q = batch_action_qs.min(axis=0)
        else:
            batch_action_q = batch_action_qs.mean(axis=0)

        pos_pool_actions = jnp.concatenate([batch['actions'][:, None, :], sampled_pos_actions], axis=1)
        pos_pool_q = jnp.concatenate([batch_action_q[:, None], sampled_pos_q], axis=1)
        pos_pool_q = jax.lax.stop_gradient(pos_pool_q)

        # Normalize Q into probabilities, then sample positives by weighted probability.
        q_min = pos_pool_q.min(axis=1, keepdims=True)
        q_max = pos_pool_q.max(axis=1, keepdims=True)
        q_norm = (pos_pool_q - q_min) / (q_max - q_min + 1e-8)
        pos_probs = jax.nn.softmax(q_norm / self.config['pos_prob_temp'], axis=1)

        sample_keys = jax.random.split(sample_pos_rng, batch_size)

        def _sample_indices(k, p):
            return jax.random.choice(k, a=p.shape[0], shape=(pos_draws,), replace=False, p=p)

        pos_idx = jax.vmap(_sample_indices)(sample_keys, pos_probs)  # [B, K]
        pos_idx_expanded = jnp.expand_dims(pos_idx, axis=-1)
        pos_idx_expanded = jnp.repeat(pos_idx_expanded, action_dim, axis=-1)
        pos_actions = jnp.take_along_axis(pos_pool_actions, pos_idx_expanded, axis=1)

        drift_prob_loss = drifting_loss(raw_actor_actions, pos_actions, temp=self.config['drift_temp'])
        drift_batch_loss = drifting_loss(
            raw_actor_actions,
            batch['actions'][:, None, :],
            temp=self.config['drift_temp'],
        )
        actor_loss = (
            self.config['drift_batch_weight'] * drift_batch_loss
            + self.config['drift_prob_weight'] * drift_prob_loss
        )

        actions = self.sample_actions(batch['observations'], seed=mse_rng)
        mse = jnp.mean((actions - batch['actions']) ** 2)

        return actor_loss, {
            'actor_loss': actor_loss,
            'bc_loss': drift_prob_loss,
            'drift_prob_loss': drift_prob_loss,
            'drift_batch_loss': drift_batch_loss,
            'pos_q': pos_pool_q.mean(),
            'pos_q_sampled': jnp.take_along_axis(pos_pool_q, pos_idx, axis=1).mean(),
            'batch_action_selected_rate': (pos_idx == 0).mean(),
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
        """Sample actions from the one-step policy."""
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

        network_info = dict(
            critic=(critic_def, (ex_observations, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_actions)),
            actor_onestep_flow=(actor_onestep_flow_def, (ex_observations, ex_actions)),
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
            agent_name='cdp_v2',  # Agent name.
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
            cql_alpha=0.1,  # Deprecated/unused in cdp_v2 critic (kept for CLI compatibility).
            cql_temp=100.0,  # Deprecated/unused in cdp_v2 critic (kept for CLI compatibility).
            cql_num_samples=8,  # Deprecated/unused in cdp_v2 critic (kept for CLI compatibility).
            drift_temp=5,  # Temperature used in drifting BC.
            drift_batch_weight=1.0,  # Weight of batch-action-only drifting loss.
            drift_prob_weight=1.0,  # Weight of probability-mined drifting loss.
            num_neg=16,  # Number of negative/generated samples per state in actor loss.
            num_samples=16,  # Number of sampled actions used to mine positive actions.
            pos_topk=2,  # Number of probability-sampled positives used in drifting loss.
            pos_prob_temp=1.0,  # Temperature for converting normalized Q values to sampling probabilities.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
        )
    )
    return config
