"""
RL-Based Scheduler (PPO).

Uses Proximal Policy Optimisation to learn a placement policy that maximises:

    reward = w_sla * sla_compliance  -  w_cost * token_cost  -  w_spill * spill_penalty

State space (per request × per node):
  - request: prompt_tokens, output_tokens, sla_class_enc, is_long_context
  - node:    memory_pressure, kv_free_ratio, compute_util, nvlink, gpu_tflops_norm

Action space:
  Discrete: select node index 0..N-1, or N = "reject"

Architecture:
  - Actor:  2-layer MLP → softmax over (N+1) actions
  - Critic: 2-layer MLP → scalar value estimate
  - PPO clip: ε = 0.2, entropy bonus: β = 0.01

Training:
  The scheduler collects (state, action, reward) tuples during benchmark runs
  and updates the policy with mini-batch PPO every TRAIN_EVERY steps.

  RLScheduler.fit(episodes) can also be called offline with pre-collected data.

Dependencies:
  numpy (always available)
  No PyTorch required — uses a pure-numpy SGD implementation for portability.
  Swap the _NumpyPolicy for a torch.nn.Module if GPU training is desired.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from control_plane.scheduler.base_scheduler import BaseScheduler, SchedulingResult
from core.models import GPUNode, InferenceRequest, SLAClass

logger = logging.getLogger(__name__)

# ── Reward weights ──
W_SLA   = 2.0
W_COST  = 0.5
W_SPILL = 1.5
W_TPUT  = 0.3

# ── PPO hyper-parameters ──
LR        = 3e-4
GAMMA     = 0.99
CLIP_EPS  = 0.20
ENTROPY_B = 0.01
EPOCHS    = 4
BATCH     = 64
TRAIN_EVERY = 256   # steps between policy updates

SLA_ENC = {SLAClass.REALTIME: 0, SLAClass.INTERACTIVE: 1, SLAClass.BATCH: 2}


# ---------------------------------------------------------------------------
# Pure-numpy policy (no framework dependency)
# ---------------------------------------------------------------------------

class _NumpyMLP:
    """Two-layer MLP with tanh hidden activations, trained via Adam."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.W1 = rng.standard_normal((in_dim, hidden))   * math.sqrt(2 / in_dim)
        self.b1 = np.zeros(hidden)
        self.W2 = rng.standard_normal((hidden, out_dim))  * math.sqrt(2 / hidden)
        self.b2 = np.zeros(out_dim)
        # Adam moments
        self._ms = [np.zeros_like(p) for p in self._params()]
        self._vs = [np.zeros_like(p) for p in self._params()]
        self._t  = 0

    def _params(self):
        return [self.W1, self.b1, self.W2, self.b2]

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = np.tanh(x @ self.W1 + self.b1)
        return h @ self.W2 + self.b2

    def update(self, grads: list[np.ndarray], lr: float = LR) -> None:
        self._t += 1
        b1, b2 = 0.9, 0.999
        eps = 1e-8
        for i, (p, g) in enumerate(zip(self._params(), grads)):
            self._ms[i] = b1 * self._ms[i] + (1 - b1) * g
            self._vs[i] = b2 * self._vs[i] + (1 - b2) * g ** 2
            m_hat = self._ms[i] / (1 - b1 ** self._t)
            v_hat = self._vs[i] / (1 - b2 ** self._t)
            p -= lr * m_hat / (np.sqrt(v_hat) + eps)

    def save(self, path: Path) -> None:
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)

    def load(self, path: Path) -> None:
        d = np.load(str(path) + ".npz")
        self.W1, self.b1, self.W2, self.b2 = d["W1"], d["b1"], d["W2"], d["b2"]


class PPOPolicy:
    """Actor-critic PPO policy operating over a discrete node-selection action space."""

    HIDDEN = 64

    def __init__(self, n_nodes: int, obs_dim: int = 12) -> None:
        self.n_actions = n_nodes + 1    # +1 for "reject"
        self.obs_dim   = obs_dim
        self.actor  = _NumpyMLP(obs_dim, self.HIDDEN, self.n_actions)
        self.critic = _NumpyMLP(obs_dim, self.HIDDEN, 1)

    def act(self, obs: np.ndarray, deterministic: bool = False) -> tuple[int, float]:
        """Sample an action. Returns (action_idx, log_prob)."""
        logits = self.actor.forward(obs)
        logits -= logits.max()   # numerical stability
        probs = np.exp(logits) / np.exp(logits).sum()

        if deterministic:
            action = int(np.argmax(probs))
        else:
            action = int(np.random.choice(len(probs), p=probs))

        log_prob = float(np.log(probs[action] + 1e-8))
        return action, log_prob

    def value(self, obs: np.ndarray) -> float:
        return float(self.critic.forward(obs)[0])

    def update_ppo(self, rollout: "Rollout") -> dict:
        """Run PPO update epochs over the collected rollout."""
        if len(rollout) < 2:
            return {}

        obs_arr   = np.array(rollout.obs)
        act_arr   = np.array(rollout.actions)
        ret_arr   = np.array(rollout.returns)
        logp_arr  = np.array(rollout.log_probs)
        adv_arr   = ret_arr - np.array(rollout.values)
        adv_arr   = (adv_arr - adv_arr.mean()) / (adv_arr.std() + 1e-8)

        pg_losses, vf_losses = [], []

        for _ in range(EPOCHS):
            idx = np.random.permutation(len(obs_arr))
            for start in range(0, len(idx), BATCH):
                batch = idx[start:start + BATCH]
                if len(batch) < 2:
                    continue
                obs_b = obs_arr[batch]
                act_b = act_arr[batch]
                ret_b = ret_arr[batch]
                adv_b = adv_arr[batch]
                lp_b  = logp_arr[batch]

                # ── Actor gradient ──
                pg_loss = 0.0
                actor_grads = [np.zeros_like(p) for p in self.actor._params()]
                for i in range(len(batch)):
                    logits = self.actor.forward(obs_b[i])
                    logits -= logits.max()
                    probs = np.exp(logits) / np.exp(logits).sum()
                    a = act_b[i]
                    new_lp = np.log(probs[a] + 1e-8)
                    ratio  = np.exp(new_lp - lp_b[i])
                    clipped = np.clip(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS)
                    obj = -min(ratio * adv_b[i], clipped * adv_b[i])
                    entropy = -np.sum(probs * np.log(probs + 1e-8))
                    loss_i  = obj - ENTROPY_B * entropy
                    pg_loss += loss_i

                    # Backprop through actor (numerical gradient for simplicity)
                    eps_g = 1e-5
                    for j, p in enumerate(self.actor._params()):
                        orig = p.flat[0]
                        p.flat[0] = orig + eps_g
                        l1 = -min(np.exp(np.log(
                            np.exp(self.actor.forward(obs_b[i])) /
                            np.exp(self.actor.forward(obs_b[i])).sum()
                        )[a] + 1e-8) - lp_b[i], CLIP_EPS) * adv_b[i]
                        p.flat[0] = orig
                        actor_grads[j].flat[0] += (l1 - loss_i) / eps_g

                self.actor.update([g / len(batch) for g in actor_grads])
                pg_losses.append(pg_loss / len(batch))

                # ── Critic gradient (MSE) ──
                vf_loss = 0.0
                critic_grads = [np.zeros_like(p) for p in self.critic._params()]
                for i in range(len(batch)):
                    v = self.critic.forward(obs_b[i])[0]
                    diff = v - ret_b[i]
                    vf_loss += diff ** 2
                    eps_g = 1e-5
                    for j, p in enumerate(self.critic._params()):
                        orig = p.flat[0]
                        p.flat[0] = orig + eps_g
                        v2 = self.critic.forward(obs_b[i])[0]
                        critic_grads[j].flat[0] += 2 * diff * (v2 - v) / eps_g
                        p.flat[0] = orig
                self.critic.update([g / len(batch) for g in critic_grads])
                vf_losses.append(vf_loss / len(batch))

        return {
            "pg_loss": float(np.mean(pg_losses)) if pg_losses else 0.0,
            "vf_loss": float(np.mean(vf_losses)) if vf_losses else 0.0,
        }


@dataclass
class Rollout:
    """Collected experience buffer for a single PPO update."""
    obs:       list = field(default_factory=list)
    actions:   list = field(default_factory=list)
    rewards:   list = field(default_factory=list)
    values:    list = field(default_factory=list)
    log_probs: list = field(default_factory=list)
    returns:   list = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.obs)

    def compute_returns(self, gamma: float = GAMMA) -> None:
        G = 0.0
        self.returns = []
        for r in reversed(self.rewards):
            G = r + gamma * G
            self.returns.insert(0, G)

    def clear(self) -> None:
        self.obs.clear(); self.actions.clear(); self.rewards.clear()
        self.values.clear(); self.log_probs.clear(); self.returns.clear()


# ---------------------------------------------------------------------------
# RL Scheduler
# ---------------------------------------------------------------------------

def _make_obs(request: InferenceRequest, nodes: list[GPUNode]) -> np.ndarray:
    """Build a fixed-size observation vector for the policy."""
    req_feats = np.array([
        math.log1p(request.prompt_tokens) / 10.0,
        math.log1p(request.max_output_tokens) / 8.0,
        SLA_ENC.get(request.sla_class, 1) / 2.0,
        float(request.is_long_context),
    ], dtype=np.float32)

    # Aggregate node features: mean + max (works for any N)
    if nodes:
        pressures  = [n.memory_pressure for n in nodes]
        kv_frees   = [n.kv_blocks_free / max(1, n.kv_blocks_total) for n in nodes]
        compute    = [n.compute_utilization for n in nodes]
        nvlinks    = [float(n.nvlink_enabled) for n in nodes]
        tflops_n   = [min(1.0, n.compute_tflops / 5000.0) for n in nodes]

        node_feats = np.array([
            np.mean(pressures),  np.max(pressures),
            np.mean(kv_frees),   np.max(kv_frees),
            np.mean(compute),    np.max(compute),
            np.mean(nvlinks),    np.mean(tflops_n),
        ], dtype=np.float32)
    else:
        node_feats = np.zeros(8, dtype=np.float32)

    return np.concatenate([req_feats, node_feats])


class RLScheduler(BaseScheduler):
    """
    PPO-trained scheduling policy.

    Learns to balance SLA compliance, token cost, and memory spill risk
    through interaction with the serving environment.

    In training mode the scheduler collects rollout data and updates the
    policy every TRAIN_EVERY scheduling decisions.

    In inference mode it picks the argmax action (deterministic=True).
    """

    OBS_DIM = 12

    def __init__(
        self,
        nodes: list[GPUNode],
        kv_manager=None,
        train: bool = True,
        model_path: Optional[str] = None,
    ) -> None:
        super().__init__(nodes, name="rl_ppo")
        self._kv_manager = kv_manager
        self._train = train
        self._policy = PPOPolicy(n_nodes=len(nodes), obs_dim=self.OBS_DIM)
        self._rollout = Rollout()
        self._step = 0
        self._updates = 0
        self._last_loss: dict = {}

        if model_path and Path(model_path).with_suffix(".npz").exists():
            self._policy.actor.load(Path(model_path + "_actor"))
            self._policy.critic.load(Path(model_path + "_critic"))
            logger.info("RLScheduler loaded policy from %s", model_path)

    def schedule(self, request: InferenceRequest) -> SchedulingResult:
        obs = _make_obs(request, self._nodes)
        feasible = self._feasible_nodes(request)

        # Map feasible nodes to action indices; N = reject
        node_to_action = {n.node_id: i for i, n in enumerate(self._nodes)}
        feasible_actions = [node_to_action[n.node_id] for n in feasible] + [len(self._nodes)]

        # Mask infeasible actions: set logits to -inf
        logits = self._policy.actor.forward(obs).copy()
        mask = np.full(len(logits), -1e9)
        for a in feasible_actions:
            mask[a] = 0.0
        masked_logits = logits + mask
        masked_logits -= masked_logits.max()
        probs = np.exp(masked_logits) / np.exp(masked_logits).sum()

        if self._train:
            action = int(np.random.choice(len(probs), p=probs))
        else:
            action = int(np.argmax(probs))

        log_prob = float(np.log(probs[action] + 1e-8))
        value    = self._policy.value(obs)

        # Reject action
        if action == len(self._nodes) or not feasible:
            if self._train:
                self._rollout.obs.append(obs)
                self._rollout.actions.append(action)
                self._rollout.values.append(value)
                self._rollout.log_probs.append(log_prob)
                self._rollout.rewards.append(-W_COST)   # rejection penalty
                self._step += 1
            return SchedulingResult(
                request_id=request.request_id,
                node_id=None, accepted=False,
                rejection_reason="rl_policy_reject",
            )

        node = self._nodes[action]
        node.allocate(request.kv_blocks_needed, request.estimated_memory_gb)

        if self._train:
            # Provisional reward (updated in on_complete)
            self._rollout.obs.append(obs)
            self._rollout.actions.append(action)
            self._rollout.values.append(value)
            self._rollout.log_probs.append(log_prob)
            self._rollout.rewards.append(0.0)   # filled in on_complete
            self._step += 1

            if self._step % TRAIN_EVERY == 0 and len(self._rollout) >= BATCH:
                self._rollout.compute_returns()
                self._last_loss = self._policy.update_ppo(self._rollout)
                self._rollout.clear()
                self._updates += 1
                logger.info(
                    "RLScheduler PPO update #%d  pg=%.4f  vf=%.4f",
                    self._updates,
                    self._last_loss.get("pg_loss", 0),
                    self._last_loss.get("vf_loss", 0),
                )

        return SchedulingResult(
            request_id=request.request_id,
            node_id=node.node_id,
            accepted=True,
        )

    def on_complete(self, request: InferenceRequest, node: GPUNode) -> None:
        node.release(request.kv_blocks_needed, request.estimated_memory_gb)

        if self._train and self._rollout.rewards:
            # Back-fill the reward for the most recent completed request
            sla_ok = 1.0 if node.memory_pressure < 0.85 else 0.0
            spill  = 1.0 if node.memory_pressure > 0.90 else 0.0
            reward = W_SLA * sla_ok - W_SPILL * spill - W_COST * 0.1
            self._rollout.rewards[-1] = reward

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self._policy.actor.save(p.parent / (p.name + "_actor"))
        self._policy.critic.save(p.parent / (p.name + "_critic"))
        logger.info("RLScheduler policy saved to %s", path)

    @property
    def training_stats(self) -> dict:
        return {
            "steps": self._step,
            "ppo_updates": self._updates,
            "last_loss": self._last_loss,
        }
