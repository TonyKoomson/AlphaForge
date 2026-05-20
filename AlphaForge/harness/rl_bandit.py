"""
AlphaForge AI Harness — RL Experiment Bandit (UCB1 + Thompson Sampling)

Learns which experiment archetypes produce higher OOS Sharpe and guides
the Strategist toward high-yield configurations.

Algorithms
----------
UCB1 (Auer, Cesa-Bianchi & Fischer, 2002):
  score(arm) = mean_reward(arm) + c * sqrt(ln(total_trials) / n(arm))

Gaussian Thompson Sampling (Chapelle & Li, 2011; Russo et al., 2018):
  θ_arm ~ N(μ_arm, σ²_arm / n_arm)   [posterior sample]
  Select argmax θ_arm.

  Thompson Sampling consistently outperforms UCB1 in the small-sample
  regime (n < 30 per arm) that characterises LLM experiment loops, where
  each trial has significant API cost.  The posterior mean and variance
  are estimated from the running sum and sum-of-squares of rewards.

Reference:
  Chapelle, O. and Li, L. (2011) 'An empirical evaluation of Thompson
  sampling', Advances in Neural Information Processing Systems, 24.

Arms represent experiment archetypes — distinct combinations of feature
categories and model complexity that have different empirical success rates.
The bandit updates its estimates after each completed iteration and persists
state across sessions via a JSON file in the harness memory directory.
"""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Optional


# ── Arm definitions ──────────────────────────────────────────────────────────

ARMS: dict[str, dict] = {
    "momentum_medium": {
        "description": "12-month + short-term price momentum, medium-depth ensemble",
        "hypothesis_hint": "Stocks with strong 12-1 month momentum continue outperforming in trending markets",
        "features": ["mom_12_1", "mom_5d", "mom_1d", "sma_50_above_200", "channel_pos_52w", "vol_21d"],
        "model_params": {"n_estimators": 150, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.8},
        "signal_threshold": 0.55,
    },
    "mean_reversion_shallow": {
        "description": "Short-term RSI/BB mean reversion with volatility filter, shallow trees",
        "hypothesis_hint": "Oversold stocks with narrowing Bollinger Bands tend to revert within 5 bars",
        "features": ["rsi_14", "bb_width", "bb_position", "vol_21d", "atr_14", "mom_5d"],
        "model_params": {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.08, "subsample": 0.9},
        "signal_threshold": 0.57,
    },
    "regime_adaptive": {
        "description": "Regime-aware features: trend + volatility + macro timing",
        "hypothesis_hint": "Combining SMA regime filter with momentum reduces whipsaw in bear markets",
        "features": ["sma_50_above_200", "sma_50_dist", "sma_200_dist", "vol_21d", "mom_12_1", "dollar_volume"],
        "model_params": {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.04, "reg_lambda": 1.5},
        "signal_threshold": 0.55,
    },
    "volume_liquidity": {
        "description": "Volume-flow and liquidity signals with momentum confirmation",
        "hypothesis_hint": "Illiquid stocks with rising dollar volume show institutional accumulation",
        "features": ["amihud_illiq", "dollar_volume", "vol_ratio_20d", "mom_5d", "rsi_14", "vol_21d"],
        "model_params": {"n_estimators": 150, "max_depth": 4, "learning_rate": 0.06},
        "signal_threshold": 0.56,
    },
    "deep_regularized": {
        "description": "All core features, deep trees, strong L2 regularization to reduce overfitting",
        "hypothesis_hint": "Combining all factor categories with regularization finds non-linear regime interactions",
        "features": ["mom_12_1", "rsi_14", "vol_21d", "sma_50_dist", "atr_14", "amihud_illiq",
                     "sma_50_above_200", "channel_pos_52w", "bb_width"],
        "model_params": {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.03,
                         "reg_lambda": 2.0, "reg_alpha": 0.5, "subsample": 0.7},
        "signal_threshold": 0.58,
    },
    "trend_conservative": {
        "description": "Pure trend-following with high signal threshold and strict SMA timing",
        "hypothesis_hint": "Only trade when SMA-200 trend + 12-month momentum align; high threshold reduces false signals",
        "features": ["sma_50_above_200", "sma_200_dist", "mom_12_1", "channel_pos_52w", "vol_21d"],
        "model_params": {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.05},
        "signal_threshold": 0.62,
    },
    "quality_growth": {
        "description": "Quality + growth proxy: high momentum with low volatility and high volume",
        "hypothesis_hint": "Quality stocks with low idiosyncratic risk and strong trend have lower drawdowns",
        "features": ["mom_12_1", "vol_21d", "dollar_volume", "sma_200_dist", "atr_14", "rsi_14"],
        "model_params": {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.04,
                         "reg_lambda": 1.0, "colsample_bytree": 0.8},
        "signal_threshold": 0.56,
    },
    "volatility_breakout": {
        "description": "Volatility contraction → breakout: BB squeeze + ATR expansion",
        "hypothesis_hint": "Low BB width followed by ATR expansion signals a volatility breakout",
        "features": ["bb_width", "atr_14", "vol_21d", "mom_5d", "channel_pos_52w", "rsi_14"],
        "model_params": {"n_estimators": 150, "max_depth": 4, "learning_rate": 0.06, "subsample": 0.8},
        "signal_threshold": 0.57,
    },
}

ARM_NAMES: list[str] = list(ARMS.keys())

# ── ExperimentBandit ──────────────────────────────────────────────────────────


class ExperimentBandit:
    """
    Thompson Sampling (default) / UCB1 multi-armed bandit for autonomous experiment type selection.

    Learns which feature/model archetypes produce higher OOS Sharpe and
    provides guided suggestions to the Strategist agent. Persists state
    between sessions so improvements accumulate across research runs.

    Parameters
    ----------
    state_path : Path
        JSON file for persisting arm statistics. Created if absent.
    exploration_c : float
        UCB1 exploration constant (used only when algorithm="ucb1").
    """

    def __init__(
        self,
        state_path: Optional[Path] = None,
        exploration_c: float = 1.0,
        algorithm: str = "thompson",
    ) -> None:
        """
        Parameters
        ----------
        state_path    : JSON file for persisting arm statistics. Created if absent.
        exploration_c : UCB1 exploration constant (used only when algorithm="ucb1").
        algorithm     : "thompson" (default) or "ucb1".
                        Thompson Sampling is preferred for small trial counts (n < 30/arm)
                        because it naturally balances exploration without a tuned constant.
        """
        if state_path is None:
            from harness.config import MEMORY_DIR
            state_path = MEMORY_DIR / "bandit_state.json"

        self.state_path    = Path(state_path)
        self.exploration_c = exploration_c
        self.algorithm     = algorithm
        self._state        = self._load()

    # ── Public API ─────────────────────────────────────────────────────────────

    def select_arm(self) -> str:
        """
        Choose the next arm using the configured algorithm.

        Returns the arm name with the highest score.
        Unvisited arms are always tried first (before any scoring).
        """
        if self.algorithm == "thompson":
            return self._select_arm_thompson()
        return self._select_arm_ucb1()

    def _select_arm_ucb1(self) -> str:
        """
        UCB1 arm selection (Auer, Cesa-Bianchi & Fischer, 2002).
        score(arm) = mean + c * sqrt(ln(N) / n)
        """
        stats = self._state["arms"]
        total = self._state["total_trials"]

        best_arm   = ARM_NAMES[0]
        best_score = -math.inf

        for arm in ARM_NAMES:
            n = stats[arm]["n"]
            if n == 0:
                return arm   # unvisited arm has infinite UCB
            mu  = stats[arm]["sum_reward"] / n
            ucb = mu + self.exploration_c * math.sqrt(math.log(max(total, 1)) / n)
            if ucb > best_score:
                best_score = ucb
                best_arm   = arm

        return best_arm

    def _select_arm_thompson(self) -> str:
        """
        Gaussian Thompson Sampling (Chapelle & Li, 2011).

        Samples arm score from the posterior distribution N(μ, σ²/n)
        where μ and σ² are estimated from observed rewards. Arms with no
        trials use a broad prior N(0, 1) to encourage exploration.

        Returns the arm with the highest sampled score.
        """
        stats = self._state["arms"]
        best_arm   = ARM_NAMES[0]
        best_sample = -math.inf

        for arm in ARM_NAMES:
            n = stats[arm]["n"]
            if n == 0:
                # Wide Gaussian prior N(0, 1) — encourages equal initial exploration
                sample = random.gauss(0.0, 1.0)
            else:
                mu = stats[arm]["sum_reward"] / n
                # Posterior variance: use sample variance / n (Gaussian conjugate prior)
                sum_sq   = stats[arm].get("sum_sq_reward", 0.0)
                variance = max(sum_sq / n - mu ** 2, 1e-6)  # clamp to avoid zero
                posterior_std = math.sqrt(variance / n)
                sample = random.gauss(mu, posterior_std)

            if sample > best_sample:
                best_sample = sample
                best_arm    = arm

        return best_arm

    def get_guidance(self, iteration: int = 1) -> dict:
        """
        Return guidance for the next experiment.

        Returns a dict with:
          arm_name        : str  — chosen archetype name
          features        : list — suggested feature list
          model_params    : dict — suggested XGBoost params
          signal_threshold: float
          rationale       : str  — human-readable explanation for Strategist
          bandit_stats    : dict — arm statistics for transparency
        """
        arm_name = self.select_arm()
        arm      = ARMS[arm_name]
        stats    = self._state["arms"][arm_name]
        total    = self._state["total_trials"]

        # Build human-readable rationale
        n      = stats["n"]
        avg_sr = (stats["sum_reward"] / n) if n > 0 else 0.0

        if n == 0:
            reason = (
                f"[Bandit] Archetype '{arm_name}' has not been tested yet. "
                "Exploring: " + arm["description"] + "."
            )
        elif avg_sr >= 0.6:
            reason = (
                f"[Bandit] Archetype '{arm_name}' has shown promise: "
                f"avg OOS Sharpe = {avg_sr:.3f} over {n} trial(s). "
                "Exploiting high-yield configuration."
            )
        elif avg_sr >= 0.3:
            reason = (
                f"[Bandit] Archetype '{arm_name}' shows partial signal: "
                f"avg OOS Sharpe = {avg_sr:.3f} over {n} trial(s). "
                "Trying refinements via bandit exploration."
            )
        else:
            reason = (
                f"[Bandit] Selected archetype '{arm_name}' for exploration "
                f"(avg OOS Sharpe = {avg_sr:.3f}, {n} trial(s), total={total})."
            )

        top3 = self._top3_summary()

        return {
            "arm_name":         arm_name,
            "description":      arm["description"],
            "hypothesis_hint":  arm["hypothesis_hint"],
            "features":         arm["features"],
            "model_params":     arm["model_params"],
            "signal_threshold": arm.get("signal_threshold", 0.55),
            "rationale":        reason,
            "top_archetypes":   top3,
        }

    def update(self, arm_name: str, oos_sharpe: float) -> None:
        """
        Record a result and update arm statistics.

        Parameters
        ----------
        arm_name : str
            The arm that was used (from get_guidance()['arm_name']).
        oos_sharpe : float
            The OOS Sharpe ratio achieved. May be negative.
        """
        if arm_name not in self._state["arms"]:
            return

        reward = float(oos_sharpe)
        arm_stats = self._state["arms"][arm_name]
        arm_stats["n"]             += 1
        arm_stats["sum_reward"]    += reward
        arm_stats["sum_sq_reward"]  = arm_stats.get("sum_sq_reward", 0.0) + reward ** 2
        arm_stats["best"]           = max(arm_stats.get("best", -999.0), reward)
        arm_stats["last"]           = reward
        self._state["total_trials"] += 1
        self._state["history"].append({
            "arm": arm_name, "reward": round(reward, 4),
            "t":   self._state["total_trials"], "ts": int(time.time()),
            "algo": self.algorithm,
        })
        self._save()

    def bootstrap_from_kb(self, kb) -> int:
        """
        Pre-load reward estimates from the knowledge base.

        Reads past experiments from the KB and infers which archetype they
        best match by comparing feature lists. Returns number of entries loaded.
        """
        try:
            entries = kb.search(query="experiment", entry_type="experiment", limit=50)
        except Exception:
            return 0

        loaded = 0
        for entry in entries:
            config  = entry.get("config", {})
            results = entry.get("results", {})
            sharpe  = float(results.get("sharpe", 0.0) or 0.0)
            feats   = set(config.get("features", []) or [])
            if not feats:
                continue
            arm = self._infer_arm(feats)
            if arm:
                self._state["arms"][arm]["n"]          += 1
                self._state["arms"][arm]["sum_reward"] += sharpe
                self._state["arms"][arm]["best"]        = max(
                    self._state["arms"][arm].get("best", -999.0), sharpe
                )
                self._state["total_trials"] += 1
                loaded += 1

        if loaded:
            self._save()
        return loaded

    def stats_summary(self) -> str:
        """Return a compact text summary of all arm statistics."""
        lines = [
            f"Bandit ({self.algorithm.upper()}): "
            f"{self._state['total_trials']} total trials across {len(ARM_NAMES)} archetypes"
        ]
        for arm in ARM_NAMES:
            s     = self._state["arms"][arm]
            n     = s["n"]
            avg   = (s["sum_reward"] / n) if n > 0 else None
            best_raw = s.get("best", None)
            best  = best_raw if (best_raw is not None and best_raw > -999) else None
            # Posterior std for Thompson — shows how confident we are about each arm
            if n >= 2 and avg is not None:
                sum_sq   = s.get("sum_sq_reward", 0.0)
                variance = max(sum_sq / n - avg ** 2, 0.0)
                post_std = math.sqrt(variance / n)
                std_str  = f"  post_std={post_std:.3f}"
            else:
                std_str = ""
            avg_str  = f"{avg:+.3f}" if avg is not None else "  n/a "
            best_str = f"{best:+.3f}" if best is not None else "  n/a "
            lines.append(
                f"  {arm:<26} n={n:2d}  avg_sr={avg_str}  best={best_str}{std_str}"
            )
        return "\n".join(lines)

    # ── Internal ───────────────────────────────────────────────────────────────

    def _load(self) -> dict:
        """Load persisted state or initialise fresh."""
        _default_arm: dict = {"n": 0, "sum_reward": 0.0, "sum_sq_reward": 0.0, "best": -999.0, "last": None}

        if self.state_path.exists():
            try:
                data = json.loads(self.state_path.read_text())
                # Ensure all current arms exist and have sum_sq_reward field
                for arm in ARM_NAMES:
                    if arm not in data["arms"]:
                        data["arms"][arm] = dict(_default_arm)
                    elif "sum_sq_reward" not in data["arms"][arm]:
                        # Backfill from existing sum_reward (assumes zero variance — conservative)
                        data["arms"][arm]["sum_sq_reward"] = 0.0
                return data
            except Exception:
                pass
        return {
            "total_trials": 0,
            "arms": {arm: dict(_default_arm) for arm in ARM_NAMES},
            "history": [],
        }

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(self._state, indent=2, default=str))

    def _infer_arm(self, features: set[str]) -> str | None:
        """Infer the closest archetype for a feature set by Jaccard similarity."""
        best_arm   = None
        best_score = 0.0
        for arm_name, arm in ARMS.items():
            arm_feats = set(arm["features"])
            overlap   = len(features & arm_feats)
            union     = len(features | arm_feats)
            jaccard   = overlap / union if union > 0 else 0.0
            if jaccard > best_score:
                best_score = jaccard
                best_arm   = arm_name
        return best_arm if best_score > 0.15 else None

    def _top3_summary(self) -> list[dict]:
        """Return top 3 arms by average reward with n > 0."""
        ranked = []
        for arm in ARM_NAMES:
            s = self._state["arms"][arm]
            if s["n"] > 0:
                avg = s["sum_reward"] / s["n"]
                ranked.append({"arm": arm, "avg_sharpe": round(avg, 3), "n": s["n"]})
        ranked.sort(key=lambda x: x["avg_sharpe"], reverse=True)
        return ranked[:3]


# ── Convenience accessor ──────────────────────────────────────────────────────

def make_bandit(
    exploration_c: float = 1.0,
    algorithm: str = "thompson",
) -> ExperimentBandit:
    """
    Factory: create bandit with default state path from config.

    Parameters
    ----------
    exploration_c : UCB1 exploration constant (ignored for Thompson Sampling)
    algorithm     : "thompson" (default, recommended) or "ucb1"
    """
    from harness.config import MEMORY_DIR
    return ExperimentBandit(
        state_path=MEMORY_DIR / "bandit_state.json",
        exploration_c=exploration_c,
        algorithm=algorithm,
    )
