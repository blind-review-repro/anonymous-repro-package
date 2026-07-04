"""CARE: Cost-Aware Resolution Routing Expert.

Implementation of Algorithm 1 from the paper. CARE is a two-stage
inference engine: feasibility filtering followed by utility maximization.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class CareKnowledgeBase:
    acc: dict
    lat: dict
    strategies: list[str]
    t_budget_feasibility: float = 70.0
    tau_art: float = 0.5
    pareto_dominators: dict = None


def care_route(kb: CareKnowledgeBase, t_budget: float, alpha: float = 1.0, beta: float = 0.5,
               degradation_severe: bool = False, degradation_kb: Optional[CareKnowledgeBase] = None) -> dict:
    if degradation_severe and degradation_kb is not None:
        active_kb = degradation_kb
    else:
        active_kb = kb

    feasible = [s for s in active_kb.strategies if active_kb.lat[s] <= t_budget]
    if not feasible:
        return {"selected": "D", "feasible": [], "rejected": {"all": "infeasible"}, "explanation": "No strategy feasible; fallback to D"}

    utilities = {s: alpha * active_kb.acc[s] - beta * (active_kb.lat[s] / t_budget) for s in feasible}
    selected = max(utilities, key=utilities.get)

    rejected = {}
    for s in active_kb.strategies:
        if s == selected:
            continue
        if s not in feasible:
            rejected[s] = f"infeasible (lat {active_kb.lat[s]:.1f} > budget {t_budget:.1f})"
        elif active_kb.pareto_dominators and s in active_kb.pareto_dominators:
            dom = active_kb.pareto_dominators[s]
            rejected[s] = f"Pareto-dominated by {dom}"
        else:
            rejected[s] = f"utility-inferior (U={utilities[s]:.4f} < {utilities[selected]:.4f})"

    explanation = (f"Selected {selected} (U={utilities[selected]:.4f}). "
                   f"Feasible: {feasible}. Rejected: " + "; ".join(f"{k}={v}" for k, v in rejected.items()))

    return {"selected": selected, "feasible": feasible, "rejected": rejected, "explanation": explanation}


if __name__ == "__main__":
    acc = {"D": 0.3304, "A": 0.4847, "B": 0.4774, "C": 0.4593}
    lat_4090 = {"D": 17.9, "A": 66.3, "B": 66.3, "C": 87.1}
    strategies = ["D", "A", "B", "C"]
    pareto = {"C": "A"}

    kb = CareKnowledgeBase(acc=acc, lat=lat_4090, strategies=strategies, pareto_dominators=pareto)

    for t_budget in [50, 100, 200]:
        result = care_route(kb, t_budget, alpha=1.0, beta=0.5)
        print(f"T_budget={t_budget}ms: {result['explanation']}")
