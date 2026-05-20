"""
AlphaForge AI Harness — Knowledge Base

Persistent JSON-backed memory for the agent research loop.
Stores experiments, discoveries, failures, and heuristics across sessions.
"""
from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from harness.config import MEMORY_DIR


class KnowledgeBase:
    """
    Lightweight knowledge base for the AI research loop.

    Entries are keyed by type:
      - experiment  : backtest/training run with parameters + results
      - finding     : human-readable insight derived from experiments
      - heuristic   : reusable rule (e.g. "high tree depth overfits on SPY")
      - failure     : documented dead-end to avoid re-exploring
      - promotion   : strategy that met the quality bar
    """

    _INDEX_FILE  = "index.json"
    _MAX_ENTRIES = 2_000

    def __init__(self, session_id: Optional[str] = None) -> None:
        self.session_id = session_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self._dir       = MEMORY_DIR
        self._dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self._dir / self._INDEX_FILE
        self._index: list[dict] = self._load_index()

    # ── Index management ──────────────────────────────────────────────────────

    def _load_index(self) -> list[dict]:
        if self._index_path.exists():
            try:
                return json.loads(self._index_path.read_text())
            except Exception:
                return []
        return []

    def _save_index(self) -> None:
        # Trim to MAX_ENTRIES (oldest first)
        if len(self._index) > self._MAX_ENTRIES:
            self._index = self._index[-self._MAX_ENTRIES:]
        self._index_path.write_text(json.dumps(self._index, indent=2))

    # ── Write ─────────────────────────────────────────────────────────────────

    def save(
        self,
        entry_type: str,
        title: str,
        body: dict | str,
        tags: Optional[list[str]] = None,
        metrics: Optional[dict] = None,
    ) -> str:
        """Persist a new knowledge entry. Returns the entry ID."""
        entry_id = str(uuid.uuid4())[:8]
        entry = {
            "id":         entry_id,
            "type":       entry_type,
            "title":      title,
            "tags":       tags or [],
            "metrics":    metrics or {},
            "session":    self.session_id,
            "timestamp":  datetime.now(timezone.utc).isoformat(),
            "body":       body,
        }
        # Save body to its own file to keep index small
        (self._dir / f"{entry_id}.json").write_text(json.dumps(entry, indent=2))
        # Add lightweight record to index
        self._index.append({
            "id":        entry_id,
            "type":      entry_type,
            "title":     title,
            "tags":      tags or [],
            "metrics":   metrics or {},
            "session":   self.session_id,
            "timestamp": entry["timestamp"],
        })
        self._save_index()
        return entry_id

    def save_experiment(
        self,
        hypothesis: str,
        config: dict,
        results: dict,
        verdict: str,
    ) -> str:
        metrics = {
            "sharpe":     results.get("sharpe", 0.0),
            "ann_return": results.get("ann_return_pct", 0.0),
            "max_dd":     results.get("max_dd_pct", 0.0),
            "oos_sharpe": results.get("oos_sharpe", results.get("sharpe", 0.0)),
        }
        return self.save(
            entry_type="experiment",
            title=hypothesis[:120],
            body={"hypothesis": hypothesis, "config": config, "results": results, "verdict": verdict},
            tags=list(config.get("features", [])),
            metrics=metrics,
        )

    def save_finding(self, title: str, insight: str, tags: Optional[list[str]] = None) -> str:
        return self.save("finding", title, {"insight": insight}, tags=tags)

    def save_heuristic(self, rule: str, reason: str, tags: Optional[list[str]] = None) -> str:
        return self.save("heuristic", rule, {"rule": rule, "reason": reason}, tags=tags)

    def save_failure(self, approach: str, reason: str, tags: Optional[list[str]] = None) -> str:
        return self.save("failure", approach, {"approach": approach, "reason": reason}, tags=tags)

    def save_promotion(self, strategy_name: str, config: dict, metrics: dict) -> str:
        return self.save(
            "promotion",
            f"PROMOTED: {strategy_name}",
            {"strategy_name": strategy_name, "config": config},
            metrics=metrics,
        )

    # ── Read ──────────────────────────────────────────────────────────────────

    def search(
        self,
        query: str = "",
        entry_type: Optional[str] = None,
        tags: Optional[list[str]] = None,
        min_sharpe: Optional[float] = None,
        limit: int = 10,
    ) -> list[dict]:
        """Return matching index entries (lightweight — no body loading)."""
        results = []
        for entry in reversed(self._index):  # newest first
            if entry_type and entry["type"] != entry_type:
                continue
            if tags and not any(t in entry.get("tags", []) for t in tags):
                continue
            if min_sharpe is not None:
                if entry.get("metrics", {}).get("sharpe", 0.0) < min_sharpe:
                    continue
            if query:
                combined = (entry.get("title", "") + " " + " ".join(entry.get("tags", []))).lower()
                if query.lower() not in combined:
                    continue
            results.append(entry)
            if len(results) >= limit:
                break
        return results

    def find_similar(
        self,
        features: list[str],
        top_n: int = 3,
        hypothesis: str = "",
    ) -> list[dict]:
        """
        Return top-N prior experiments that are semantically similar to the proposed
        feature list and/or hypothesis text.

        Two similarity signals are fused:
          1. Jaccard similarity of feature tag sets (structural overlap)
          2. TF-IDF cosine similarity of hypothesis text (semantic overlap)
             Falls back to Jaccard-only if sklearn is unavailable.

        The fused score = 0.6 * jaccard + 0.4 * tfidf_cosine (when text available).
        Threshold: only report entries with fused score > 0.30.
        """
        q_set = set(features) if features else set()
        experiments = [e for e in self._index if e.get("type") == "experiment"]
        if not experiments:
            return []

        # ── TF-IDF cosine similarity on hypothesis text ──────────────────────
        tfidf_scores: dict[str, float] = {}
        if hypothesis.strip():
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                import math as _math

                titles = [e.get("title", "") for e in experiments]
                corpus = [hypothesis] + titles
                vec    = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
                tfidf  = vec.fit_transform(corpus).toarray()
                q_vec  = tfidf[0]
                q_norm = _math.sqrt(sum(x * x for x in q_vec)) or 1.0
                for i, entry in enumerate(experiments):
                    d_vec  = tfidf[i + 1]
                    d_norm = _math.sqrt(sum(x * x for x in d_vec)) or 1.0
                    cosine = sum(a * b for a, b in zip(q_vec, d_vec)) / (q_norm * d_norm)
                    tfidf_scores[entry["id"]] = cosine
            except ImportError:
                pass  # sklearn not available — use Jaccard only

        # ── Fuse scores ───────────────────────────────────────────────────────
        scored = []
        for entry in experiments:
            e_tags = set(entry.get("tags", []))
            if q_set and e_tags:
                jaccard = len(q_set & e_tags) / len(q_set | e_tags)
            elif q_set or e_tags:
                jaccard = 0.0
            else:
                jaccard = 0.0

            tfidf_cos = tfidf_scores.get(entry["id"], 0.0)

            if tfidf_scores:
                fused = 0.6 * jaccard + 0.4 * tfidf_cos
            else:
                fused = jaccard

            if fused > 0.30:
                scored.append((fused, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:top_n]]

    def get(self, entry_id: str) -> Optional[dict]:
        """Load full entry body by ID."""
        path = self._dir / f"{entry_id}.json"
        if path.exists():
            try:
                return json.loads(path.read_text())
            except Exception:
                return None
        return None

    def get_promotions(self) -> list[dict]:
        """Return all promoted strategies with full body."""
        index_entries = self.search(entry_type="promotion", limit=50)
        return [e for e in (self.get(e["id"]) for e in index_entries) if e]

    def get_failures(self) -> list[dict]:
        return [e for e in (self.get(e["id"]) for e in self.search(entry_type="failure", limit=30)) if e]

    def get_best_experiments(self, n: int = 5) -> list[dict]:
        """Return the top-N experiments by OOS Sharpe."""
        exps = self.search(entry_type="experiment", limit=200)
        exps.sort(key=lambda e: e.get("metrics", {}).get("oos_sharpe", 0.0), reverse=True)
        return [e for e in (self.get(e["id"]) for e in exps[:n]) if e]

    # ── Summary for agents ────────────────────────────────────────────────────

    def context_summary(self, max_chars: int = 3_000, bandit=None) -> str:
        """
        Compact text summary of the knowledge base for injection into agent prompts.
        Includes: promotions, top experiments, recent failures, key heuristics,
        and (optionally) RL bandit top archetypes.

        Parameters
        ----------
        bandit : ExperimentBandit | None
            If provided, the top-performing archetypes are appended so the
            Strategist can exploit the best-learned experiment types.
        """
        lines: list[str] = []

        promotions = self.get_promotions()
        if promotions:
            lines.append("## Promoted Strategies")
            for p in promotions[-3:]:
                b = p.get("body", {})
                m = p.get("metrics", {})
                lines.append(
                    f"- {b.get('strategy_name','?')}: Sharpe={m.get('sharpe','?'):.2f}, "
                    f"DD={m.get('max_dd','?'):.1f}%"
                )

        best = self.get_best_experiments(5)
        if best:
            lines.append("\n## Best Experiments (by OOS Sharpe)")
            for exp in best:
                b = exp.get("body", {})
                m = exp.get("metrics", {})
                lines.append(
                    f"- [{exp.get('metrics',{}).get('oos_sharpe',0):.2f} Sharpe] "
                    f"{b.get('hypothesis','')[:80]}  "
                    f"features={b.get('config',{}).get('features',[])}"
                )

        failures = self.get_failures()
        if failures:
            lines.append("\n## Known Dead-Ends (avoid re-exploring)")
            for f in failures[-5:]:
                b = f.get("body", {})
                lines.append(f"- {b.get('approach','?')}: {b.get('reason','?')}")

        heuristics = [e for e in (self.get(e["id"]) for e in self.search(entry_type="heuristic", limit=10)) if e]
        if heuristics:
            lines.append("\n## Established Heuristics")
            for h in heuristics[-5:]:
                b = h.get("body", {})
                lines.append(f"- {b.get('rule','?')} (reason: {b.get('reason','?')})")

        # Inject RL bandit intelligence if provided
        if bandit is not None:
            try:
                state = bandit._state.get("arms", {})
                visited = {a: s for a, s in state.items() if s.get("n", 0) > 0}
                if visited:
                    ranked = sorted(
                        visited.items(),
                        key=lambda kv: kv[1]["sum_reward"] / kv[1]["n"],
                        reverse=True,
                    )
                    lines.append("\n## RL Bandit — Top Archetypes (by avg OOS Sharpe)")
                    for arm_name, s in ranked[:3]:
                        avg = s["sum_reward"] / s["n"]
                        lines.append(
                            f"- {arm_name}: avg Sharpe={avg:+.3f} over {s['n']} trial(s)"
                        )
                    bottom = ranked[-1] if len(ranked) > 1 else None
                    if bottom:
                        avg_b = bottom[1]["sum_reward"] / bottom[1]["n"]
                        lines.append(
                            f"  (avoid: {bottom[0]} avg Sharpe={avg_b:+.3f})"
                        )
            except Exception:
                pass

        summary = "\n".join(lines)
        if len(summary) > max_chars:
            summary = summary[:max_chars] + "\n[...truncated]"
        return summary or "No prior experiments. This is the first research session."

    def stats(self) -> dict:
        counts: dict[str, int] = {}
        for e in self._index:
            counts[e["type"]] = counts.get(e["type"], 0) + 1
        return {"total": len(self._index), "by_type": counts}
