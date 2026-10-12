from typing import List, Dict, Any, Optional, Tuple
from ..models.schemas import CandidateAudit
from .circuit_breaker import CircuitBreaker
from ..chaos.manager import ChaosManager
from ..storage.telemetry import TelemetryStore

def evaluate_candidates(
    models: List[Dict[str, Any]],
    prompt_tokens: int,
    expected_output_tokens: int,
    remaining_latency_ms: float,
    remaining_cost_usd: float,
    required_capability: float,
    required_context: Optional[int],
    circuit_breakers: Dict[str, CircuitBreaker],
    chaos_manager: ChaosManager,
    telemetry: TelemetryStore
) -> Tuple[List[CandidateAudit], Optional[str]]:
    """
    Hard Constraint Engine.
    Rules decide. AI only advises.
    Every candidate model is strictly evaluated against remaining budgets and thresholds.
    Generates transparent rejection reasons and closest-miss recommendations.
    """
    total_tokens = prompt_tokens + expected_output_tokens
    audits: List[CandidateAudit] = []
    valid_candidates: List[CandidateAudit] = []

    closest_miss: Optional[str] = None
    min_miss_distance = float('inf')

    for m in models:
        reasons: List[str] = []
        m_id = m["id"]

        # 1. Circuit Breaker
        cb = circuit_breakers.get(m_id)
        is_healthy = cb.allow_request() if cb else True
        if not is_healthy:
            reasons.append(f"Circuit breaker OPEN (consecutive failures: {cb.consecutive_failures if cb else 0})")

        # 2. Context Window
        effective_context_req = max(total_tokens, required_context or 0)
        context_fit = m["context_window"] >= effective_context_req
        if not context_fit:
            reasons.append(f"Context window exceeded ({m['context_window']} < required {effective_context_req})")

        # 3. Cost Budget
        est_cost = (prompt_tokens / 1000.0 * m["input_cost_per_1k"]) + (expected_output_tokens / 1000.0 * m["output_cost_per_1k"])
        cost_fit = est_cost <= remaining_cost_usd
        if not cost_fit:
            reasons.append(f"Cost ${est_cost:.6f} exceeds remaining budget ${remaining_cost_usd:.6f}")

        # 4. Latency Budget (Calibrated SLA + Live EMA + Chaos Injection)
        if m.get("endpoint_type") == "local_gguf":
            measured_lat = m["base_latency_ms"]
        else:
            measured_lat = telemetry.get_rolling_p95_latency(m_id) or m["base_latency_ms"]
        injected_lat = chaos_manager.get_injected_latency(m_id)
        predicted_lat = measured_lat + injected_lat
        latency_fit = predicted_lat <= remaining_latency_ms
        if not latency_fit:
            reasons.append(f"Latency {predicted_lat:.0f}ms exceeds remaining budget {remaining_latency_ms:.0f}ms")

        # 5. Capability
        cap_fit = m["capability_score"] >= required_capability
        if not cap_fit:
            reasons.append(f"Capability {m['capability_score']:.2f} < required {required_capability:.2f}")

        # 6. Endpoint Type Check (Demo mode: local models only, exclude cloud API from routing)
        is_cloud_api = (m.get("endpoint_type") == "cloud_api")
        if is_cloud_api:
            reasons.append("Cloud API inactive (Demo mode: local models only)")

        passed = not is_cloud_api and is_healthy and context_fit and cost_fit and latency_fit and cap_fit

        audit = CandidateAudit(
            model_id=m_id,
            model_name=m["name"],
            passed=passed,
            rejection_reasons=reasons,
            context_fit=context_fit,
            predicted_cost_usd=round(est_cost, 6),
            cost_fit=cost_fit,
            predicted_latency_ms=round(predicted_lat, 1),
            latency_fit=latency_fit,
            capability_score=m["capability_score"],
            capability_fit=cap_fit,
            circuit_state=cb.state if cb else "CLOSED",
            score=0.0
        )
        audits.append(audit)

        if passed:
            valid_candidates.append(audit)
        else:
            # Calculate distance for Closest Miss suggestion
            cost_gap = max(0.0, est_cost - remaining_cost_usd) / (remaining_cost_usd if remaining_cost_usd > 0 else 1.0)
            lat_gap = max(0.0, predicted_lat - remaining_latency_ms) / (remaining_latency_ms if remaining_latency_ms > 0 else 1.0)
            cap_gap = max(0.0, required_capability - m["capability_score"]) / (required_capability if required_capability > 0 else 1.0)
            
            gap = cost_gap + lat_gap + cap_gap
            if gap < min_miss_distance:
                min_miss_distance = gap
                if lat_gap > 0 and cost_gap == 0 and cap_gap == 0:
                    diff_ms = int(predicted_lat - remaining_latency_ms)
                    closest_miss = f"Feasible on {m['name']} if max latency is raised by {diff_ms}ms"
                elif cost_gap > 0 and lat_gap == 0 and cap_gap == 0:
                    diff_cost = est_cost - remaining_cost_usd
                    closest_miss = f"Feasible on {m['name']} if max cost is raised by ${diff_cost:.5f}"
                elif cap_gap > 0 and cost_gap == 0 and lat_gap == 0:
                    closest_miss = f"Feasible on {m['name']} if min capability is relaxed to {m['capability_score']:.2f}"
                else:
                    closest_miss = f"Close to feasible on {m['name']} with slightly higher latency/budget"

    return audits, closest_miss
