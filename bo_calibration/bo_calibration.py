#!/usr/bin/env python3
"""Scenario-conditional alpha/beta calibration for CARE via Bayesian Optimization.

Oracle-reviewed design:
- (alpha, beta) are scenario-conditional, not global
- Objective = deployment success rate (accuracy target AND latency budget met)
- 3 deployment scenarios: edge-constrained, cloud-quality, real-time-bursty
- BO with GP + EI, 30 iterations, 5 random initial points
- Baselines: fixed (1.0, 0.5), linear heuristic (1.0, Lat_ratio), random
- Ablation: remove strategy C (SR@1280) to test structure sensitivity
- What-if: 4090-trained (alpha,beta) applied to 5060 latencies

Usage:
    python scripts/bo_calibration.py
Outputs:
    eval_results/bo_calibration.json
    figures/bo_convergence.pdf
    figures/bo_scenario_comparison.pdf
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

ACC = {"D": 0.3304, "A": 0.4847, "B": 0.4774, "C": 0.4593}
LAT_4090 = {"D": 17.9, "A": 66.3, "B": 66.3, "C": 87.1}
LAT_5060 = {"D": 14.9, "A": 41.2, "B": 41.2, "C": 233.6}

STRATEGIES_FULL = ["D", "A", "B", "C"]
STRATEGIES_NO_C = ["D", "A", "B"]


@dataclass
class Scenario:
    label: str
    t_budget_sampler: Callable[[], float]
    acc_target: float
    description: str


SCENARIOS = [
    Scenario(
        label="edge-constrained",
        t_budget_sampler=lambda: float(np.random.uniform(15, 80)),
        acc_target=0.30,
        description="Edge device with tight latency budget; tolerate lower accuracy",
    ),
    Scenario(
        label="cloud-quality",
        t_budget_sampler=lambda: float(np.random.uniform(150, 350)),
        acc_target=0.45,
        description="Cloud GPU with generous budget; prioritize accuracy",
    ),
    Scenario(
        label="real-time-bursty",
        t_budget_sampler=lambda: float(
            np.random.choice([np.random.uniform(30, 60), np.random.uniform(100, 200)], p=[0.6, 0.4])
        ),
        acc_target=0.40,
        description="Real-time surveillance with bursty budget; mixed regime",
    ),
]


def care_route(alpha: float, beta: float, t_budget: float, lat: dict, strategies: list[str]) -> str:
    """CARE two-stage inference: feasibility filter + utility maximization."""
    feasible = [s for s in strategies if lat[s] <= t_budget]
    if not feasible:
        return "D"  # fallback to lowest-latency
    utilities = {s: alpha * ACC[s] - beta * (lat[s] / t_budget) for s in feasible}
    return max(utilities, key=utilities.get)


def deployment_success_rate(alpha: float, beta: float, scenario: Scenario, lat: dict, strategies: list[str], n_traces: int = 500) -> float:
    """Fraction of traces where CARE's choice meets both accuracy target AND latency budget."""
    successes = 0
    for _ in range(n_traces):
        t_budget = scenario.t_budget_sampler()
        chosen = care_route(alpha, beta, t_budget, lat, strategies)
        # Success: chosen strategy is feasible (latency <= budget) AND accuracy meets target
        if lat[chosen] <= t_budget and ACC[chosen] >= scenario.acc_target:
            successes += 1
    return successes / n_traces


def oracle_upper_bound(scenario: Scenario, lat: dict, strategies: list[str], n_traces: int = 500) -> float:
    """Brute-force best strategy per trace (upper bound on success rate)."""
    successes = 0
    for _ in range(n_traces):
        t_budget = scenario.t_budget_sampler()
        # Oracle: pick any feasible strategy that meets accuracy target
        feasible_ok = [s for s in strategies if lat[s] <= t_budget and ACC[s] >= scenario.acc_target]
        if feasible_ok:
            successes += 1
    return successes / n_traces


# ---- Bayesian Optimization (GP + EI) ----
# Minimal implementation to avoid scikit-optimize dependency.


def gp_predict(X_train, y_train, X_test, length_scale=0.3, noise=1e-6):
    """Gaussian process regression with RBF kernel over 2D (alpha, beta)."""
    X_tr = np.asarray(X_train, dtype=float)
    X_te = np.asarray(X_test, dtype=float)
    def sq_dist(P, Q):
        return np.sum(P ** 2, axis=1)[:, None] + np.sum(Q ** 2, axis=1)[None, :] - 2 * P @ Q.T
    K = np.exp(-sq_dist(X_tr, X_tr) / (2 * length_scale ** 2))
    K += noise * np.eye(len(X_tr))
    K_s = np.exp(-sq_dist(X_te, X_tr) / (2 * length_scale ** 2))
    try:
        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
        mu = K_s @ alpha
        v = np.linalg.solve(L, K_s.T)
        sigma = np.sqrt(np.maximum(np.diag(K_s @ K_s.T) - np.sum(v * v, axis=0), 1e-10))
        return mu, sigma
    except np.linalg.LinAlgError:
        return np.full(len(X_te), np.mean(y_train)), np.full(len(X_te), 1.0)


def expected_improvement(mu, sigma, y_best, xi=0.01):
    """EI acquisition function."""
    sigma = np.maximum(sigma, 1e-10)
    imp = mu - y_best - xi
    Z = imp / sigma
    ei = imp * (0.5 * (1 + np.vectorize(math.erf)(Z / math.sqrt(2))))
    return ei


def bayesian_optimize(objective: Callable[[float, float], float], bounds=((0.1, 2.0), (0.1, 2.0)), n_init=5, n_iter=30):
    """Maximize objective over (alpha, beta) using GP + EI."""
    X_train = []
    y_train = []
    # Random initial points
    for _ in range(n_init):
        a = float(np.random.uniform(*bounds[0]))
        b = float(np.random.uniform(*bounds[1]))
        X_train.append([a, b])
        y_train.append(objective(a, b))

    convergence = [max(y_train)]

    for _ in range(n_iter):
        # Candidate pool
        candidates = np.array([[a, b] for a in np.linspace(*bounds[0], 20) for b in np.linspace(*bounds[1], 20)])
        mu, sigma = gp_predict(X_train, y_train, candidates)
        y_best = max(y_train)
        ei = expected_improvement(mu, sigma, y_best)
        next_idx = int(np.argmax(ei))
        next_pt = candidates[next_idx]
        y_new = objective(next_pt[0], next_pt[1])
        X_train.append(list(next_pt))
        y_train.append(y_new)
        convergence.append(max(y_train))

    best_idx = int(np.argmax(y_train))
    return {"alpha": X_train[best_idx][0], "beta": X_train[best_idx][1], "success_rate": y_train[best_idx], "convergence": convergence}


def linear_heuristic(scenario: Scenario, lat: dict, strategies: list[str], n_traces: int = 500) -> float:
    """Oracle's suggested baseline: alpha=1, beta=Lat(C)/Lat(A) ratio."""
    # Heuristic: beta scales with SR cost relative to LR-resize
    beta_heuristic = lat["C"] / lat["A"] if "C" in strategies else 0.5
    beta_heuristic = min(max(beta_heuristic, 0.1), 2.0)
    return deployment_success_rate(1.0, beta_heuristic, scenario, lat, strategies, n_traces)


def run_experiments():
    results = {"scenarios": {}, "hardware": {"4090": LAT_4090, "5060": LAT_5060}}

    for hw_label, lat in [("4090", LAT_4090), ("5060", LAT_5060)]:
        for scenario in SCENARIOS:
            key = f"{hw_label}_{scenario.label}"
            print(f"\n=== {key} ===")

            # Oracle upper bound
            oracle = oracle_upper_bound(scenario, lat, STRATEGIES_FULL)
            print(f"  Oracle upper bound: {oracle:.3f}")

            # Fixed baseline (current CARE)
            fixed = deployment_success_rate(1.0, 0.5, scenario, lat, STRATEGIES_FULL)
            print(f"  Fixed (1.0, 0.5): {fixed:.3f}")

            # Linear heuristic baseline
            lin_heur = linear_heuristic(scenario, lat, STRATEGIES_FULL)
            print(f"  Linear heuristic: {lin_heur:.3f}")

            # BO
            def obj(a, b, sc=scenario, l=lat):
                return deployment_success_rate(a, b, sc, l, STRATEGIES_FULL)

            bo = bayesian_optimize(obj)
            print(f"  BO: alpha={bo['alpha']:.3f}, beta={bo['beta']:.3f}, success={bo['success_rate']:.3f}")

            # Ablation: remove strategy C
            oracle_no_c = oracle_upper_bound(scenario, lat, STRATEGIES_NO_C)
            bo_no_c_obj = lambda a, b, sc=scenario, l=lat: deployment_success_rate(a, b, sc, l, STRATEGIES_NO_C)
            bo_no_c = bayesian_optimize(bo_no_c_obj)
            print(f"  BO (no C): alpha={bo_no_c['alpha']:.3f}, beta={bo_no_c['beta']:.3f}, success={bo_no_c['success_rate']:.3f}")

            results["scenarios"][key] = {
                "scenario": scenario.label,
                "hardware": hw_label,
                "description": scenario.description,
                "acc_target": scenario.acc_target,
                "oracle_upper_bound": round(oracle, 4),
                "fixed_1_0.5": round(fixed, 4),
                "linear_heuristic": round(lin_heur, 4),
                "bo_alpha": round(bo["alpha"], 4),
                "bo_beta": round(bo["beta"], 4),
                "bo_success_rate": round(bo["success_rate"], 4),
                "bo_convergence": [round(c, 4) for c in bo["convergence"]],
                "ablation_no_c_oracle": round(oracle_no_c, 4),
                "ablation_no_c_bo_alpha": round(bo_no_c["alpha"], 4),
                "ablation_no_c_bo_beta": round(bo_no_c["beta"], 4),
                "ablation_no_c_bo_success": round(bo_no_c["success_rate"], 4),
            }

    return results


def plot_convergence(results, out_path):
    """Plot BO convergence for 3 scenarios on 4090."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    scenario_labels = ["edge-constrained", "cloud-quality", "real-time-bursty"]
    for i, sc in enumerate(scenario_labels):
        key = f"4090_{sc}"
        conv = results["scenarios"][key]["bo_convergence"]
        fixed = results["scenarios"][key]["fixed_1_0.5"]
        oracle = results["scenarios"][key]["oracle_upper_bound"]
        ax = axes[i]
        ax.plot(range(len(conv)), conv, "o-", color="#6a4c93", markersize=4, label="BO best so far")
        ax.axhline(y=fixed, color="#1f77b4", linestyle="--", label="Fixed (1.0, 0.5)")
        ax.axhline(y=oracle, color="#2ca02c", linestyle=":", label="Oracle upper bound")
        ax.set_title(sc, fontsize=10)
        ax.set_xlabel("BO iteration", fontsize=9)
        ax.set_ylabel("Success rate", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_scenario_comparison(results, out_path):
    """Bar chart: success rate by scenario and method on 4090."""
    scenario_labels = ["edge-constrained", "cloud-quality", "real-time-bursty"]
    methods = ["fixed_1_0.5", "linear_heuristic", "bo_success_rate", "oracle_upper_bound"]
    method_labels = ["Fixed (1.0, 0.5)", "Linear heuristic", "BO-calibrated", "Oracle"]

    x = np.arange(len(scenario_labels))
    width = 0.2
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (m, ml) in enumerate(zip(methods, method_labels)):
        vals = [results["scenarios"][f"4090_{sc}"][m] for sc in scenario_labels]
        ax.bar(x + i * width, vals, width, label=ml)
    ax.set_ylabel("Deployment success rate", fontsize=10)
    ax.set_title("CARE calibration on RTX 4090 across deployment scenarios", fontsize=11)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(scenario_labels, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    out_dir = Path("/mnt/e/dengjiahao/video-sr/eval_results")
    fig_dir = Path("/mnt/e/dengjiahao/video-sr/paper/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Running scenario-conditional BO calibration...")
    results = run_experiments()

    out_json = out_dir / "bo_calibration.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_json}")

    plot_convergence(results, fig_dir / "bo_convergence.pdf")
    plot_scenario_comparison(results, fig_dir / "bo_scenario_comparison.pdf")

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY (4090)")
    print("=" * 80)
    print(f"{'Scenario':<22} {'Fixed':>7} {'LinHeur':>8} {'BO':>7} {'Oracle':>7} {'BO(a,b)':>14}")
    for sc in ["edge-constrained", "cloud-quality", "real-time-bursty"]:
        r = results["scenarios"][f"4090_{sc}"]
        print(f"{sc:<22} {r['fixed_1_0.5']:>7.3f} {r['linear_heuristic']:>8.3f} {r['bo_success_rate']:>7.3f} {r['oracle_upper_bound']:>7.3f}   ({r['bo_alpha']:.2f},{r['bo_beta']:.2f})")


if __name__ == "__main__":
    main()
