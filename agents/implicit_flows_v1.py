from functools import partial
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, ValueVectorField


class ImplicitFlowsV1Agent(flax.struct.PyTreeNode):
    """Implicit Flows v1 agent (no one-step flow, RS-only extraction)."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def critic_loss(self, batch, grad_params, rng):
        """Compute implicit critic loss (kept from implicit flows)."""
        batch_size = batch['actions'].shape[0]
        rng, actor_rng, noise_rng, time_rng, q_rng, ret_rng = jax.random.split(rng, 6)

        # Keep Value Flows style action extraction for the next action.
        next_actions = self.sample_actions(batch['next_observations'], actor_rng)

        times = jax.random.uniform(time_rng, (batch_size, 1))
        next_noises = jax.random.normal(noise_rng, (batch_size, 1))
        noisy_next_returns1, ret_jac_eps_prods1 = self.compute_flow_returns(
            next_noises,
            batch['next_observations'],
            next_actions,
            end_times=times,
            flow_network_name='target_critic_flow1',
            return_jac_eps_prod=True,
        )
        noisy_next_returns2, ret_jac_eps_prods2 = self.compute_flow_returns(
            next_noises,
            batch['next_observations'],
            next_actions,
            end_times=times,
            flow_network_name='target_critic_flow2',
            return_jac_eps_prod=True,
        )

        if self.config['ret_agg'] == 'min':
            noisy_next_returns_bellman = jnp.minimum(noisy_next_returns1, noisy_next_returns2)
        elif self.config['ret_agg'] == 'mean':
            noisy_next_returns_bellman = (
                noisy_next_returns1 + noisy_next_returns2
            ) / 2
        else:
            raise ValueError(f"Invalid ret_agg: {self.config['ret_agg']}")

        noises = jax.random.normal(ret_rng, (batch_size, 1))
        r_noises = noises - self.config['discount'] * jnp.expand_dims(batch['masks'], axis=-1) * next_noises
        r_vector_field = jnp.expand_dims(batch['rewards'], axis=-1) - r_noises
        
        noisy_returns1, ret_jac_eps_prods1 = self.compute_flow_returns(
            next_noises,
            batch['next_observations'],
            next_actions,
            end_times=times,
            flow_network_name='target_critic_flow1',
            return_jac_eps_prod=True,
        )
        noisy_returns2, ret_jac_eps_prods2 = self.compute_flow_returns(
            next_noises,
            batch['next_observations'],
            next_actions,
            end_times=times,
            flow_network_name='target_critic_flow2',
            return_jac_eps_prod=True,
        )
        ret_stds1 = jnp.sqrt(ret_jac_eps_prods1 ** 2)
        ret_stds2 = jnp.sqrt(ret_jac_eps_prods2 ** 2)

        ret_stds = 0.5 * (ret_stds1 + ret_stds2)
        if self.config['q_agg'] == 'min':
            ret_stds = jnp.minimum(ret_stds1, ret_stds2)
        else:
            ret_stds = (ret_stds1 + ret_stds2) / 2
        weights = jax.nn.sigmoid(-self.config['confidence_weight_temp'] / ret_stds) + 0.5
        weights = jax.lax.stop_gradient(weights)

        noisy_returns = (noisy_returns1 + noisy_returns2) / 2

        vector_field1 = self.network.select('critic_flow1')(
            noisy_returns, times, batch['observations'], batch['actions'], params=grad_params
        )
        vector_field2 = self.network.select('critic_flow2')(
            noisy_returns, times, batch['observations'], batch['actions'], params=grad_params
        )
        next_vector_field1 = self.network.select('target_critic_flow1')(
            noisy_next_returns_bellman, times, batch['next_observations'], next_actions
        )
        next_vector_field2 = self.network.select('target_critic_flow2')(
            noisy_next_returns_bellman, times, batch['next_observations'], next_actions
        )

        if self.config['ret_agg'] == 'min':
            mixed_next_vector_field = jnp.minimum(next_vector_field1, next_vector_field2)
        elif self.config['ret_agg'] == 'mean':
            mixed_next_vector_field = (next_vector_field1 + next_vector_field2) / 2
        else:
            raise ValueError(f"Invalid ret_agg: {self.config['ret_agg']}")

        target_vector_field = self.config['discount'] * jnp.expand_dims(batch['masks'], axis=-1) * mixed_next_vector_field + r_vector_field
        implicit_loss = ((vector_field1 - target_vector_field) ** 2 + (vector_field2 - target_vector_field) ** 2).mean(axis=-1)
        critic_loss = (weights * implicit_loss).mean()

        q_noises = jax.random.normal(q_rng, (batch_size, 1))
        q1 = (q_noises + self.network.select('critic_flow1')(
            q_noises, jnp.zeros_like(q_noises), batch['observations'], batch['actions'])).squeeze(-1)
        q2 = (q_noises + self.network.select('critic_flow2')(
            q_noises, jnp.zeros_like(q_noises), batch['observations'], batch['actions'])).squeeze(-1)
        if self.config['clip_flow_returns']:
            q1 = jnp.clip(
                q1,
                self.config['min_reward'] / (1 - self.config['discount']),
                self.config['max_reward'] / (1 - self.config['discount']),
            )
            q2 = jnp.clip(
                q2,
                self.config['min_reward'] / (1 - self.config['discount']),
                self.config['max_reward'] / (1 - self.config['discount']),
            )
        if self.config['q_agg'] == 'min':
            q = jnp.minimum(q1, q2)
        else:
            q = (q1 + q2) / 2

        return critic_loss, {
            'critic_loss': critic_loss,
            'implicit_loss': implicit_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    def actor_loss(self, batch, grad_params, rng):
        """Actor loss = alpha * bc_flow_loss + q_loss(proxy_action)."""
        batch_size, action_dim = batch['actions'].shape
        rng, x_rng, t_rng, q_rng, mse_rng = jax.random.split(rng, 5)

        # BC flow loss.
        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch['actions']
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network.select('actor_flow')(batch['observations'], x_t, t, params=grad_params)
        bc_flow_loss = jnp.mean((pred - vel) ** 2)

        # Proxy action: one-step Euler from t to 1, stopping gradients through x_t.
        x_t_sg = jax.lax.stop_gradient(x_t)
        t_sg = t
        proxy_vector_field = self.network.select('actor_flow')(
            batch['observations'], x_t_sg, t_sg, params=grad_params
        )
        proxy_actions = x_t_sg + (1.0 - t_sg) * proxy_vector_field
        proxy_actions = jnp.clip(proxy_actions, -1, 1)

        # Critic Q through proxy actions.
        q_noises = jax.random.normal(q_rng, (batch_size, 1))
        q_times = jnp.zeros_like(q_noises)
        q1 = (q_noises + self.network.select('critic_flow1')(
            q_noises, q_times, batch['observations'], proxy_actions)).squeeze(-1)
        q2 = (q_noises + self.network.select('critic_flow2')(
            q_noises, q_times, batch['observations'], proxy_actions)).squeeze(-1)
        if self.config['clip_flow_returns']:
            q1 = jnp.clip(
                q1,
                self.config['min_reward'] / (1 - self.config['discount']),
                self.config['max_reward'] / (1 - self.config['discount']),
            )
            q2 = jnp.clip(
                q2,
                self.config['min_reward'] / (1 - self.config['discount']),
                self.config['max_reward'] / (1 - self.config['discount']),
            )
        if self.config['q_agg'] == 'min':
            q = jnp.minimum(q1, q2)
        else:
            q = (q1 + q2) / 2

        q_loss = -q.mean()
        if self.config['normalize_q_loss']:
            lam = jax.lax.stop_gradient(1 / jnp.abs(q).mean())
            q_loss = lam * q_loss

        actor_loss = self.config['alpha'] * bc_flow_loss + q_loss

        actions = self.sample_actions(
            batch['observations'],
            seed=mse_rng,
        )
        mse = jnp.mean((actions - batch['actions']) ** 2)

        return actor_loss, {
            'actor_loss': actor_loss,
            'bc_flow_loss': bc_flow_loss,
            'q_loss': q_loss,
            'q': q.mean(),
            'mse': mse,
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute total loss."""
        info = {}
        rng = rng if rng is not None else self.rng
        rng, critic_rng, actor_rng = jax.random.split(rng, 3)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = critic_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        """Update target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch):
        """Update the agent and return info."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'critic_flow1')
        self.target_update(new_network, 'critic_flow2')

        return self.replace(network=new_network, rng=new_rng), info

    @partial(jax.jit, static_argnames=('flow_network_name', 'return_jac_eps_prod'))
    def compute_flow_returns(
        self,
        noises,
        observations,
        actions,
        init_times=None,
        end_times=None,
        flow_network_name='critic_flow',
        return_jac_eps_prod=False,
    ):
        """Compute returns from the return flow model with Euler integration."""
        noisy_returns = noises
        noisy_jac_eps_prod = jnp.ones_like(noises)
        if init_times is None:
            init_times = jnp.zeros((*noisy_returns.shape[:-1], 1), dtype=noisy_returns.dtype)
        if end_times is None:
            end_times = jnp.ones((*noisy_returns.shape[:-1], 1), dtype=noisy_returns.dtype)
        step_size = (end_times - init_times) / self.config['num_flow_steps']

        def func(carry, i):
            noisy_returns, noisy_jac_eps_prod = carry

            times = i * step_size + init_times
            vector_field, jac_eps_prod = jax.jvp(
                lambda ret: self.network.select(flow_network_name)(ret, times, observations, actions),
                (noisy_returns,),
                (noisy_jac_eps_prod,),
            )

            new_noisy_returns = noisy_returns + step_size * vector_field
            new_noisy_jac_eps_prod = noisy_jac_eps_prod + step_size * jac_eps_prod
            if self.config['clip_flow_returns']:
                new_noisy_returns = jnp.clip(
                    new_noisy_returns,
                    self.config['min_reward'] / (1 - self.config['discount']),
                    self.config['max_reward'] / (1 - self.config['discount']),
                )

            return (new_noisy_returns, new_noisy_jac_eps_prod), None

        (noisy_returns, noisy_jac_eps_prod), _ = jax.lax.scan(
            func,
            (noisy_returns, noisy_jac_eps_prod),
            jnp.arange(self.config['num_flow_steps']),
        )

        if return_jac_eps_prod:
            return noisy_returns, noisy_jac_eps_prod
        return noisy_returns

    @jax.jit
    def compute_flow_actions(
        self,
        noises,
        observations,
        params=None,
        init_times=None,
        end_times=None,
    ):
        """Compute actions from BC flow with Euler integration."""
        noisy_actions = noises
        if init_times is None:
            init_times = jnp.zeros((*noisy_actions.shape[:-1], 1), dtype=noisy_actions.dtype)
        if end_times is None:
            end_times = jnp.ones((*noisy_actions.shape[:-1], 1), dtype=noisy_actions.dtype)
        step_size = (end_times - init_times) / self.config['num_flow_steps']

        def func(carry, i):
            (noisy_actions,) = carry

            times = i * step_size + init_times
            vector_field = self.network.select('actor_flow')(
                observations, noisy_actions, times, params=params
            )
            new_noisy_actions = noisy_actions + vector_field * step_size
            if self.config['clip_flow_actions']:
                new_noisy_actions = jnp.clip(new_noisy_actions, -1, 1)

            return (new_noisy_actions,), None

        (noisy_actions,), _ = jax.lax.scan(
            func,
            (noisy_actions,),
            jnp.arange(self.config['num_flow_steps']),
        )

        if not self.config['clip_flow_actions']:
            noisy_actions = jnp.clip(noisy_actions, -1, 1)
        return noisy_actions

    @partial(jax.jit, static_argnames=('policy_extraction'))
    def sample_actions(
        self,
        observations,
        seed=None,
        temperature=1.0,
        policy_extraction='rs',
    ):
        """Sample actions with full-step actor flow (rejection sampling only)."""
        del temperature
        if policy_extraction != 'rs':
            raise ValueError(
                f"implicit_flows_v1 only supports policy_extraction='rs', got {policy_extraction}"
            )

        action_seed, q_seed = jax.random.split(seed)
        actor_noises = jax.random.normal(
            action_seed,
            (
                *observations.shape[: -len(self.config['ob_dims'])],
                self.config['num_samples'],
                self.config['action_dim'],
            ),
        )

        n_observations = jnp.repeat(
            jnp.expand_dims(observations, -2),
            self.config['num_samples'],
            axis=-2,
        )
        actions = self.compute_flow_actions(actor_noises, n_observations)

        q_noises = jax.random.normal(
            q_seed,
            (*observations.shape[: -len(self.config['ob_dims'])], self.config['num_samples'], 1),
        )
        q1 = (
            q_noises
            + self.network.select('critic_flow1')(
                q_noises, jnp.zeros_like(q_noises), n_observations, actions
            )
        ).squeeze(-1)
        q2 = (
            q_noises
            + self.network.select('critic_flow2')(
                q_noises, jnp.zeros_like(q_noises), n_observations, actions
            )
        ).squeeze(-1)
        if self.config['clip_flow_returns']:
            q1 = jnp.clip(
                q1,
                self.config['min_reward'] / (1 - self.config['discount']),
                self.config['max_reward'] / (1 - self.config['discount']),
            )
            q2 = jnp.clip(
                q2,
                self.config['min_reward'] / (1 - self.config['discount']),
                self.config['max_reward'] / (1 - self.config['discount']),
            )
        if self.config['q_agg'] == 'min':
            q = jnp.minimum(q1, q2)
        else:
            q = (q1 + q2) / 2

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
        """Create a new agent."""
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_observations = example_batch['observations']
        ex_actions = example_batch['actions']
        ex_returns = ex_actions[..., :1]
        ex_times = ex_actions[..., :1]
        ob_dims = ex_observations.shape[1:]
        action_dim = ex_actions.shape[-1]
        min_reward = example_batch['min_reward']
        max_reward = example_batch['max_reward']

        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic_flow'] = encoder_module()
            encoders['target_critic_flow'] = encoder_module()
            encoders['actor_flow'] = encoder_module()

        critic_flow1_def = ValueVectorField(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['value_layer_norm'],
            num_ensembles=1,
            encoder=encoders.get('critic_flow'),
        )
        critic_flow2_def = ValueVectorField(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['value_layer_norm'],
            num_ensembles=1,
            encoder=encoders.get('critic_flow'),
        )
        target_critic_flow1_def = ValueVectorField(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['value_layer_norm'],
            num_ensembles=1,
            encoder=encoders.get('target_critic_flow'),
        )
        target_critic_flow2_def = ValueVectorField(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['value_layer_norm'],
            num_ensembles=1,
            encoder=encoders.get('target_critic_flow'),
        )
        actor_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_flow'),
        )

        network_info = dict(
            critic_flow1=(critic_flow1_def, (ex_returns, ex_times, ex_observations, ex_actions)),
            critic_flow2=(critic_flow2_def, (ex_returns, ex_times, ex_observations, ex_actions)),
            target_critic_flow1=(target_critic_flow1_def, (ex_returns, ex_times, ex_observations, ex_actions)),
            target_critic_flow2=(target_critic_flow2_def, (ex_returns, ex_times, ex_observations, ex_actions)),
            actor_flow=(actor_flow_def, (ex_observations, ex_actions, ex_times)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network_params
        params['modules_target_critic_flow1'] = params['modules_critic_flow1']
        params['modules_target_critic_flow2'] = params['modules_critic_flow2']

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim
        config['min_reward'] = min_reward
        config['max_reward'] = max_reward
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='implicit_flows_v1',
            ob_dims=ml_collections.config_dict.placeholder(list),
            action_dim=ml_collections.config_dict.placeholder(int),
            min_reward=ml_collections.config_dict.placeholder(float),
            max_reward=ml_collections.config_dict.placeholder(float),
            lr=3e-4,
            batch_size=256,
            actor_hidden_dims=(512, 512, 512, 512),
            value_hidden_dims=(512, 512, 512, 512),
            actor_layer_norm=True,
            value_layer_norm=True,
            discount=0.99,
            tau=0.005,
            ret_agg='mean',  # 'min', 'mean', 'adaptive_addq', or soft weighting.
            q_agg='mean',
            clip_flow_actions=True,
            clip_flow_returns=True,
            addq_low_threshold=0.75,
            addq_high_threshold=1.25,
            addq_beta_low=0.25,
            addq_beta_mid=0.5,
            addq_beta_high=0.75,
            addq_eps=1e-8,
            num_samples=16,
            num_flow_steps=10,
            normalize_q_loss=True,
            confidence_weight_temp=0.3,  # Temperature for the confidence weights.
            alpha=10.0,
            encoder=ml_collections.config_dict.placeholder(str),
        )
    )
    return config
