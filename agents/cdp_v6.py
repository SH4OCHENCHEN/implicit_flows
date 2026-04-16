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
    pos_logit_bias=None,
    neg_logit_bias=None,
):
    """Compute drift field V with explicit positive and negative sets.

    Args:
        gen: Generated samples [..., G, D]
        pos: Positive samples [..., P, D]
        neg: Negative samples [..., N, D]
        temp: Temperature for softmax kernel.
        exclude_self_neg: Whether to mask diagonal in gen-neg distances.
        pos_logit_bias: Optional bias added to positive logits, broadcastable to [..., G, P].
        neg_logit_bias: Optional bias added to negative logits, broadcastable to [..., G, N].

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
    if pos_logit_bias is not None:
        logits_pos = logits_pos + pos_logit_bias
    if neg_logit_bias is not None:
        logits_neg = logits_neg + neg_logit_bias
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
    pos_logit_bias=None,
    neg_logit_bias=None,
):
    """Drifting loss: MSE(gen, stopgrad(gen + V))."""
    v = compute_drift(
        gen,
        pos,
        neg,
        temp=temp,
        exclude_self_neg=exclude_self_neg,
        pos_logit_bias=pos_logit_bias,
        neg_logit_bias=neg_logit_bias,
    )
    target = jax.lax.stop_gradient(gen + v)
    return jnp.mean((gen - target) ** 2)


def multi_temp_drifting_loss(
    gen: jnp.ndarray,
    pos: jnp.ndarray,
    neg: jnp.ndarray,
    temps,
    exclude_self_neg: bool = False,
    pos_logit_bias=None,
    neg_logit_bias=None,
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
            pos_logit_bias=pos_logit_bias,
            neg_logit_bias=neg_logit_bias,
        )
    return loss


def select_action_subsets(actions: jnp.ndarray, scores: jnp.ndarray, topk: int, bottomk: int):
    """Select top/bottom actions based on per-action scores.

    Args:
        actions: Candidate actions [B, N, A].
        scores: Candidate scores [B, N].
        topk: Number of highest-scoring actions to keep.
        bottomk: Number of lowest-scoring actions to keep.

    Returns:
        top_actions: [B, topk', A]
        bottom_actions: [B, bottomk', A]
        top_scores: [B, topk']
        bottom_scores: [B, bottomk']
    """
    pool_size = actions.shape[1]
    topk = min(max(int(topk), 0), pool_size)
    bottomk = min(max(int(bottomk), 0), pool_size - topk)

    sorted_idx = jnp.argsort(scores, axis=1)
    bottom_idx = sorted_idx[:, :bottomk]
    top_idx = sorted_idx[:, pool_size - topk:] if topk > 0 else sorted_idx[:, :0]

    action_dim = actions.shape[-1]
    top_idx_expanded = jnp.repeat(jnp.expand_dims(top_idx, axis=-1), action_dim, axis=-1)
    bottom_idx_expanded = jnp.repeat(jnp.expand_dims(bottom_idx, axis=-1), action_dim, axis=-1)

    top_actions = jnp.take_along_axis(actions, top_idx_expanded, axis=1)
    bottom_actions = jnp.take_along_axis(actions, bottom_idx_expanded, axis=1)
    top_scores = jnp.take_along_axis(scores, top_idx, axis=1)
    bottom_scores = jnp.take_along_axis(scores, bottom_idx, axis=1)
    return top_actions, bottom_actions, top_scores, bottom_scores


class CDPV6Agent(flax.struct.PyTreeNode):
    """CDP v6 agent with mixed positive/negative pools for policy drift.

    Diff vs `cdp_v5`:
    - Policy positives are MPPI-refined behavior proposals only (no dataset perturb positives).
    - Policy positive logits are Q-biased on the MPPI positive pool.
    - Policy negatives are mixed: policy self + behavior low-Q actions.
    - Middle-Q behavior candidates are dropped from policy drift.
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

    def _score_action_candidates(self, observations, candidate_actions, grad_params):
        """Score a batch of candidate actions with the aggregated critic Q."""
        batch_size, num_actions, action_dim = candidate_actions.shape
        candidate_obs_repeat = jnp.repeat(observations[:, None, ...], num_actions, axis=1)
        candidate_obs_flat = candidate_obs_repeat.reshape(
            (batch_size * num_actions, *observations.shape[1:])
        )
        candidate_actions_flat = candidate_actions.reshape((batch_size * num_actions, action_dim))

        candidate_qs = self.network.select('critic')(
            candidate_obs_flat, actions=candidate_actions_flat, params=grad_params
        )
        if self.config['q_agg'] == 'min':
            candidate_q = candidate_qs.min(axis=0)
        else:
            candidate_q = candidate_qs.mean(axis=0)
        candidate_q = candidate_q.reshape((batch_size, num_actions))
        return jax.lax.stop_gradient(candidate_q)

    def _mppi_refine_behavior_actions(self, observations, behavior_candidate_actions, grad_params, rng):
        """Refine behavior proposals with a stop-gradient MPPI/CEM-style update."""
        refined_actions = jax.lax.stop_gradient(behavior_candidate_actions)
        q_dtype = behavior_candidate_actions.dtype
        weight_entropy = jnp.asarray(0.0, dtype=q_dtype)
        std_mean = jnp.asarray(0.0, dtype=q_dtype)

        if self.config['mppi_enable']:
            for _ in range(int(self.config['mppi_iters'])):
                rng, sample_rng = jax.random.split(rng)
                candidate_q = self._score_action_candidates(observations, refined_actions, grad_params)
                shifted_q = candidate_q - jnp.max(candidate_q, axis=1, keepdims=True)
                weights = jax.nn.softmax(shifted_q / self.config['mppi_temp'], axis=1)
                weights = jax.lax.stop_gradient(weights)

                mean = jnp.sum(weights[..., None] * refined_actions, axis=1, keepdims=True)
                mean = jax.lax.stop_gradient(mean)

                centered_actions = refined_actions - mean
                var = jnp.sum(weights[..., None] * jnp.square(centered_actions), axis=1, keepdims=True)
                std = jnp.sqrt(jnp.maximum(var, 1e-8))
                std = jnp.clip(
                    jnp.maximum(std, self.config['mppi_std']),
                    self.config['mppi_std_min'],
                    self.config['mppi_std_max'],
                )
                std = jax.lax.stop_gradient(std)

                noise = jax.random.normal(sample_rng, refined_actions.shape)
                refined_actions = mean + std * noise
                refined_actions = jnp.clip(refined_actions, -1.0, 1.0)
                refined_actions = jax.lax.stop_gradient(refined_actions)

                weight_entropy = (-weights * jnp.log(weights + 1e-12)).sum(axis=1).mean()
                weight_entropy = jax.lax.stop_gradient(weight_entropy)
                std_mean = std.mean()
                std_mean = jax.lax.stop_gradient(std_mean)

        refined_q = self._score_action_candidates(observations, refined_actions, grad_params)
        return refined_actions, refined_q, weight_entropy, std_mean

    def actor_loss(self, batch, grad_params, rng):
        """Train behavior/policy actors with drifting losses and different positive sets."""
        batch_size, action_dim = batch['actions'].shape
        behavior_num_neg = self.config['behavior_num_neg']
        policy_num_neg = self.config['policy_num_neg']
        behavior_pool_size = self.config['mppi_num_samples']
        behavior_bottomk_neg = self.config['behavior_bottomk_neg']
        drift_temps = self.config['drift_temps']
        mppi_num_pos = self.config['mppi_num_pos']
        _, behavior_noise_rng, policy_x_noise_rng, behavior_pool_noise_rng, mse_rng, mppi_rng = (
            jax.random.split(rng, 6)
        )

        # =========================
        # 1) Behavior actor loss:
        # positives = dataset action only (one positive)
        # =========================
        behavior_noises = jax.random.normal(behavior_noise_rng, (batch_size, behavior_num_neg, action_dim))
        behavior_obs_repeat = jnp.repeat(batch['observations'][:, None, ...], behavior_num_neg, axis=1)
        behavior_obs_flat = behavior_obs_repeat.reshape(
            (batch_size * behavior_num_neg, *batch['observations'].shape[1:])
        )
        behavior_noises_flat = behavior_noises.reshape((batch_size * behavior_num_neg, action_dim))

        raw_behavior_actions_flat = self.network.select('actor_behavior_onestep_flow')(
            behavior_obs_flat, behavior_noises_flat, params=grad_params
        )
        raw_behavior_actions = raw_behavior_actions_flat.reshape((batch_size, behavior_num_neg, action_dim))
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
        # positives = MPPI-refined behavior top-Q only
        # negatives = policy self + behavior low-Q
        # =========================
        policy_neg_obs_repeat = jnp.repeat(batch['observations'][:, None, ...], policy_num_neg, axis=1)
        policy_neg_obs_flat = policy_neg_obs_repeat.reshape(
            (batch_size * policy_num_neg, *batch['observations'].shape[1:])
        )

        policy_x_noises = jax.random.normal(policy_x_noise_rng, (batch_size, policy_num_neg, action_dim))
        policy_x_noises_flat = policy_x_noises.reshape((batch_size * policy_num_neg, action_dim))
        raw_policy_actions_flat = self.network.select('actor_onestep_flow')(
            policy_neg_obs_flat, policy_x_noises_flat, params=grad_params
        )
        raw_policy_actions = raw_policy_actions_flat.reshape((batch_size, policy_num_neg, action_dim))

        # Build a larger behavior candidate pool and score with critic.
        behavior_pool_noises = jax.random.normal(
            behavior_pool_noise_rng, (batch_size, behavior_pool_size, action_dim)
        )
        behavior_pool_obs_repeat = jnp.repeat(
            batch['observations'][:, None, ...], behavior_pool_size, axis=1
        )
        behavior_pool_obs_flat = behavior_pool_obs_repeat.reshape(
            (batch_size * behavior_pool_size, *batch['observations'].shape[1:])
        )
        behavior_pool_noises_flat = behavior_pool_noises.reshape((batch_size * behavior_pool_size, action_dim))
        behavior_pool_actions_flat = self.network.select('actor_behavior_onestep_flow')(
            behavior_pool_obs_flat, behavior_pool_noises_flat
        )
        behavior_pool_actions_flat = jnp.clip(behavior_pool_actions_flat, -1, 1)
        behavior_pool_actions = behavior_pool_actions_flat.reshape(
            (batch_size, behavior_pool_size, action_dim)
        )

        behavior_pool_q = self._score_action_candidates(
            batch['observations'], behavior_pool_actions, grad_params=None,
        )
        refined_behavior_actions, refined_behavior_q, mppi_weight_entropy, mppi_std_mean = (
            self._mppi_refine_behavior_actions(
                batch['observations'],
                behavior_pool_actions,
                grad_params=None,
                rng=mppi_rng,
            )
        )

        pos_behavior_actions, _, pos_behavior_q, _ = select_action_subsets(
            refined_behavior_actions,
            refined_behavior_q,
            topk=mppi_num_pos,
            bottomk=0,
        )
        _, neg_behavior_actions, _, neg_behavior_q = select_action_subsets(
            behavior_pool_actions,
            behavior_pool_q,
            topk=0,
            bottomk=behavior_bottomk_neg,
        )

        pos_actions = pos_behavior_actions
        neg_actions = jnp.concatenate([raw_policy_actions, neg_behavior_actions], axis=1)

        # Build per-positive weights from state-wise scaled exp(Q) on the mixed positive pool.
        if pos_actions.shape[1] > 0:
            policy_pos_obs_repeat = jnp.repeat(batch['observations'][:, None, ...], pos_actions.shape[1], axis=1)
            policy_pos_obs_flat = policy_pos_obs_repeat.reshape(
                (batch_size * pos_actions.shape[1], *batch['observations'].shape[1:])
            )
            pos_actions_flat = pos_actions.reshape((batch_size * pos_actions.shape[1], action_dim))

            pos_qs = self.network.select('critic')(
                policy_pos_obs_flat, actions=pos_actions_flat
            )
            if self.config['q_agg'] == 'min':
                pos_q = pos_qs.min(axis=0)
            else:
                pos_q = pos_qs.mean(axis=0)
            pos_q = pos_q.reshape((batch_size, pos_actions.shape[1]))
            pos_q = jax.lax.stop_gradient(pos_q)

            lam = jax.lax.stop_gradient(1.0 / (jnp.abs(pos_q).mean(axis=1, keepdims=True) + 1e-8))
            pos_q_scaled = pos_q * lam
            pos_probs = jax.nn.softmax(pos_q_scaled / self.config['pos_prob_temp'], axis=1)
            pos_probs = jax.lax.stop_gradient(pos_probs)
            pos_logit_bias = jnp.expand_dims(jnp.log(pos_probs + 1e-12), axis=1)

            pos_q_mean = pos_q.mean()
            pos_q_scaled_mean = pos_q_scaled.mean()
            pos_q_weighted_mean = (pos_probs * pos_q).sum(axis=1).mean()
            lam_mean = lam.mean()
            lam_min = lam.min()
            lam_max = lam.max()
            pos_entropy = (-pos_probs * jnp.log(pos_probs + 1e-12)).sum(axis=1).mean()
        else:
            pos_logit_bias = None
            pos_q_mean = jnp.asarray(0.0, dtype=behavior_pool_q.dtype)
            pos_q_scaled_mean = jnp.asarray(0.0, dtype=behavior_pool_q.dtype)
            pos_q_weighted_mean = jnp.asarray(0.0, dtype=behavior_pool_q.dtype)
            lam_mean = jnp.asarray(0.0, dtype=behavior_pool_q.dtype)
            lam_min = jnp.asarray(0.0, dtype=behavior_pool_q.dtype)
            lam_max = jnp.asarray(0.0, dtype=behavior_pool_q.dtype)
            pos_entropy = jnp.asarray(0.0, dtype=behavior_pool_q.dtype)

        policy_drift_loss = multi_temp_drifting_loss(
            raw_policy_actions,
            pos_actions,
            neg_actions,
            temps=drift_temps,
            exclude_self_neg=True,
            pos_logit_bias=pos_logit_bias,
        )

        # Total actor loss: train both actors together.
        actor_loss = (
            self.config['drift_batch_weight'] * behavior_drift_loss
            + self.config['drift_prob_weight'] * policy_drift_loss
        )

        if pos_behavior_q.shape[1] > 0:
            pos_behavior_q_mean = pos_behavior_q.mean()
        else:
            pos_behavior_q_mean = jnp.asarray(0.0, dtype=behavior_pool_q.dtype)

        if neg_behavior_q.shape[1] > 0:
            neg_behavior_q_mean = neg_behavior_q.mean()
        else:
            neg_behavior_q_mean = jnp.asarray(0.0, dtype=behavior_pool_q.dtype)

        behavior_q_gap = pos_behavior_q_mean - neg_behavior_q_mean

        # MSE is measured on policy actor outputs.
        actions = self.sample_actions(batch['observations'], seed=mse_rng)
        mse = jnp.mean((actions - batch['actions']) ** 2)

        return actor_loss, {
            'actor_loss': actor_loss,
            'behavior_drift_loss': behavior_drift_loss,
            'policy_drift_loss': policy_drift_loss,
            'drift_batch_loss': behavior_drift_loss,  # backward-compatible key
            'drift_prob_loss': policy_drift_loss,  # backward-compatible key
            'pos_q': pos_q_mean,  # backward-compatible key
            'behavior_pool_q_mean': behavior_pool_q.mean(),
            'behavior_pool_q_max': behavior_pool_q.max(),
            'behavior_pool_q_min': behavior_pool_q.min(),
            'mppi_pos_q_mean': pos_behavior_q_mean,
            'mppi_pos_q_max': pos_behavior_q.max() if pos_behavior_q.shape[1] > 0 else jnp.asarray(0.0, dtype=behavior_pool_q.dtype),
            'mppi_weight_entropy': mppi_weight_entropy,
            'mppi_std_mean': mppi_std_mean,
            'pos_actions_q_mean': pos_q_mean,
            'pos_q_scaled': pos_q_scaled_mean,
            'pos_q_weighted': pos_q_weighted_mean,
            'lam_mean': lam_mean,
            'lam_min': lam_min,
            'lam_max': lam_max,
            'pos_entropy': pos_entropy,
            'pos_behavior_q_mean': pos_behavior_q_mean,
            'neg_behavior_q_mean': neg_behavior_q_mean,
            'behavior_q_gap': behavior_q_gap,
            'drift_temp_count': jnp.asarray(len(drift_temps)),
            'behavior_num_neg': jnp.asarray(behavior_num_neg),
            'behavior_pool_size': jnp.asarray(behavior_pool_size),
            'behavior_topk_pos': jnp.asarray(min(max(int(mppi_num_pos), 0), behavior_pool_size)),
            'behavior_bottomk_neg': jnp.asarray(
                min(
                    max(int(behavior_bottomk_neg), 0),
                    behavior_pool_size,
                )
            ),
            'dataset_num_pos': jnp.asarray(0),
            'policy_pos_total': jnp.asarray(pos_actions.shape[1]),
            'policy_neg_total': jnp.asarray(neg_actions.shape[1]),
            'policy_num_pos': jnp.asarray(mppi_num_pos),  # backward-compatible key
            'policy_num_neg': jnp.asarray(policy_num_neg),
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
        del temperature
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
            agent_name='cdp_v6',  # Agent name.
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
            drift_prob_weight=1.0,  # Weight of policy-actor drifting loss (mixed positive/negative pools).
            behavior_num_neg=16,  # Number of generated negatives for behavior drifting.
            behavior_bottomk_neg=4,  # Number of low-Q behavior candidates used as policy negatives.
            mppi_enable=True,  # Whether to refine behavior proposals before selecting policy positives.
            mppi_iters=3,  # Number of MPPI refinement iterations on the behavior proposal pool.
            mppi_num_samples=16,  # Number of MPPI refinement samples / behavior proposal candidates.
            mppi_num_pos=16,  # Number of refined MPPI behavior actions used as policy positives.
            mppi_temp=1.0,  # Temperature for MPPI proposal reweighting.
            mppi_std=0.10,  # Base/floor std for MPPI resampling.
            mppi_std_min=0.03,  # Minimum diagonal std for MPPI resampling.
            mppi_std_max=0.30,  # Maximum diagonal std for MPPI resampling.
            policy_num_neg=16,  # Number of policy self-generated negatives for policy drifting.
            pos_prob_temp=10,  # Temperature for exp(Q) sampling probabilities on mixed policy positives.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
        )
    )
    return config
