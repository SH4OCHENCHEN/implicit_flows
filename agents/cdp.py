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

    kernel = jnp.exp(-dist / temp)

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


class CDPAgent(flax.struct.PyTreeNode):
    """Conservative drifting policy agent."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def critic_loss(self, batch, grad_params, rng):
        """Compute the TD critic loss with a conservative OOD penalty."""
        rng, sample_rng, cql_rng = jax.random.split(rng, 3)
        next_actions = self.sample_actions(batch['next_observations'], seed=sample_rng)
        next_actions = jnp.clip(next_actions, -1, 1)

        next_qs = self.network.select('target_critic')(batch['next_observations'], actions=next_actions)
        if self.config['q_agg'] == 'min':
            next_q = next_qs.min(axis=0)
        else:
            next_q = next_qs.mean(axis=0)

        target_q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_q

        q = self.network.select('critic')(batch['observations'], actions=batch['actions'], params=grad_params)
        bellman_loss = jnp.square(q - target_q).mean()

        policy_actions = self.sample_actions(batch['observations'], seed=cql_rng)
        policy_actions = jnp.clip(policy_actions, -1, 1)
        q_pi = self.network.select('critic')(batch['observations'], actions=policy_actions, params=grad_params)
        q_beta = self.network.select('critic')(batch['observations'], actions=batch['actions'], params=grad_params)
        ood_penalty = self.config['cql_alpha'] * ((q_pi - q_beta).mean())

        critic_loss = bellman_loss + ood_penalty

        return critic_loss, {
            'critic_loss': critic_loss,
            'bellman_loss': bellman_loss,
            'ood_penalty': ood_penalty,
            'q_pi_mean': q_pi.mean(),
            'q_beta_mean': q_beta.mean(),
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    def actor_loss(self, batch, grad_params, rng):
        """Compute the drifting-policy actor loss."""
        batch_size, action_dim = batch['actions'].shape
        num_neg = self.config['num_neg']
        rng, noise_rng = jax.random.split(rng)

        # [B, G, D]: generate multiple raw actor actions per state.
        noises = jax.random.normal(noise_rng, (batch_size, num_neg, action_dim))
        obs_repeat = jnp.repeat(batch['observations'][:, None, ...], num_neg, axis=1)
        obs_flat = obs_repeat.reshape((batch_size * num_neg, *batch['observations'].shape[1:]))
        noises_flat = noises.reshape((batch_size * num_neg, action_dim))

        raw_actions_flat = self.network.select('actor_onestep_flow')(
            obs_flat, noises_flat, params=grad_params
        )
        raw_actor_actions = raw_actions_flat.reshape((batch_size, num_neg, action_dim))

        # Use BxGxD and Bx1xD directly in drifting loss so V has shape BxGxD.
        pos_actions = batch['actions'][:, None, :]
        bc_loss = drifting_loss(raw_actor_actions, pos_actions, temp=self.config['drift_temp'])

        actor_actions = jnp.clip(raw_actions_flat, -1, 1)
        qs = self.network.select('critic')(obs_flat, actions=actor_actions)
        q = jnp.mean(qs, axis=0)

        q_loss = -q.mean()
        if self.config['normalize_q_loss']:
            lam = jax.lax.stop_gradient(1 / jnp.abs(q).mean())
            q_loss = lam * q_loss

        actor_loss = self.config['alpha'] * bc_loss #+ q_loss

        actions = self.sample_actions(batch['observations'], seed=rng)
        mse = jnp.mean((actions - batch['actions']) ** 2)

        return actor_loss, {
            'actor_loss': actor_loss,
            'bc_loss': bc_loss,
            'q_loss': q_loss,
            'q': q.mean(),
            'mse': mse,
            'pos_gap': jnp.linalg.norm(raw_actor_actions - pos_actions, axis=-1).mean(),
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
        """Sample actions with rejection sampling from one-step policy."""
        del temperature
        num_samples = self.config['num_samples'] if 'num_samples' in self.config else self.config['num_neg']
        action_seed = seed
        actor_noises = jax.random.normal(
            action_seed,
            (
                *observations.shape[: -len(self.config['ob_dims'])],
                num_samples,
                self.config['action_dim'],
            ),
        )
        n_observations = jnp.repeat(
            jnp.expand_dims(observations, -2),
            num_samples,
            axis=-2,
        )
        actions = self.network.select('actor_onestep_flow')(n_observations, actor_noises)
        actions = jnp.clip(actions, -1, 1)

        qs = self.network.select('critic')(n_observations, actions=actions)
        if self.config['q_agg'] == 'min':
            q = qs.min(axis=0)
        else:
            q = qs.mean(axis=0)

        if len(q.shape) > 1:
            actions = actions[jnp.arange(q.shape[0]), jnp.argmax(q, axis=-1)]
        else:
            actions = actions[jnp.argmax(q, axis=-1)]
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
            agent_name='cdp',  # Agent name.
            ob_dims=ml_collections.config_dict.placeholder(list),  # Observation dimensions (will be set automatically).
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (will be set automatically).
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            q_agg='mean',  # Aggregation method for target Q values.
            alpha=10.0,  # Actor BC coefficient.
            cql_alpha=0.0,  # Conservative critic coefficient.
            drift_temp=50,  # Temperature used in drifting BC.
            num_neg=16,  # Number of negative/generated samples per state in actor loss.
            num_samples=16,  # Number of sampled actions for rejection sampling.
            normalize_q_loss=True,  # Whether to normalize the Q loss.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
        )
    )
    return config
