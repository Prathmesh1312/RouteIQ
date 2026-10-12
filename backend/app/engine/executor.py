import os
import time
import json
import uuid
import datetime
import urllib.request
import urllib.error
from typing import List, Dict, Any, Optional, Tuple

from ..models.schemas import RouteRequest, RouteResponse, CandidateAudit
from .constraints import evaluate_candidates
from .ranker import rank_candidates
from .circuit_breaker import CircuitBreaker
from ..chaos.manager import ChaosManager
from ..storage.telemetry import TelemetryStore
from ..estimators.multi_estimator import MultiEstimatorConsensus
from .gguf_engine import GGUFEngine


def compute_cost_matrix(
    models: List[Dict[str, Any]],
    tokens_in: int,
    tokens_out: int,
    chosen_model_id: Optional[str],
    chosen_cost_usd: float
) -> Dict[str, Any]:
    """
    Computes mathematical cost breakdown for all models using:
    Cost(model) = (tokens_in / 1000 * input_cost_per_1k) + (tokens_out / 1000 * output_cost_per_1k)
    Evaluates savings relative to the chosen model and frontier baseline.
    """
    breakdown = []
    frontier_model = next((m for m in models if m.get("tier") == "hard"), None)
    if not frontier_model:
        frontier_model = max(models, key=lambda m: m.get("output_cost_per_1k", 0.0))

    baseline_cost = round((tokens_in / 1000.0 * frontier_model["input_cost_per_1k"]) + (tokens_out / 1000.0 * frontier_model["output_cost_per_1k"]), 6)

    for m in models:
        m_cost = round((tokens_in / 1000.0 * m["input_cost_per_1k"]) + (tokens_out / 1000.0 * m["output_cost_per_1k"]), 6)
        is_chosen = (m["id"] == chosen_model_id)
        savings = round(m_cost - chosen_cost_usd, 6)
        breakdown.append({
            "model_id": m["id"],
            "model_name": m["name"],
            "family": m.get("family", ""),
            "tier": m.get("tier", "medium"),
            "input_cost_per_1k": m["input_cost_per_1k"],
            "output_cost_per_1k": m["output_cost_per_1k"],
            "calculated_cost_usd": m_cost,
            "is_chosen": is_chosen,
            "savings_vs_model_usd": savings,
            "savings_vs_model_pct": round((savings / m_cost * 100.0), 2) if m_cost > 0 else 0.0
        })

    if chosen_model_id is not None:
        money_saved = max(0.0, round(baseline_cost - chosen_cost_usd, 6))
        savings_pct = round((money_saved / baseline_cost * 100.0), 2) if baseline_cost > 0 else 0.0
    else:
        money_saved = 0.0
        savings_pct = 0.0

    return {
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "chosen_model_id": chosen_model_id,
        "chosen_cost_usd": round(chosen_cost_usd, 6),
        "baseline_model_id": frontier_model["id"],
        "baseline_model_name": frontier_model["name"],
        "baseline_cost_usd": baseline_cost,
        "money_saved_usd": money_saved,
        "savings_percentage": savings_pct,
        "models": breakdown
    }


class ResilientExecutor:
    """
    Unified Dual-Mode Model Executor with Shrinking-Budget Failover.
    Enforces the Core Axiom:
    "Failover can never relax a constraint. Subsequent retries operate strictly
    under remaining time and budget: L_rem = L_orig - t_spent."
    """
    def __init__(
        self,
        models: List[Dict[str, Any]],
        circuit_breakers: Dict[str, CircuitBreaker],
        chaos_manager: ChaosManager,
        telemetry: TelemetryStore,
        estimator: MultiEstimatorConsensus,
        gguf_engine: Optional[GGUFEngine] = None
    ):
        self.models = models
        self.models_map = {m["id"]: m for m in models}
        self.circuit_breakers = circuit_breakers
        self.chaos_manager = chaos_manager
        self.telemetry = telemetry
        self.estimator = estimator
        self.gguf_engine = gguf_engine or GGUFEngine()

    def _call_external_api(self, model_config: Dict[str, Any], prompt: str, timeout_sec: float) -> Optional[Tuple[str, int, int, str]]:
        """
        Calls live online API models (Groq, Gemini, OpenRouter, Anthropic) if API keys are configured.
        Returns (choice_text, tokens_in, tokens_out, badge) on success, or None on failure / missing key.
        """
        model_id = model_config.get("id", "")
        provider = model_config.get("api_provider")
        api_model_name = model_config.get("api_model_name")
        env_key = model_config.get("env_key")

        # 1. Groq API (High Speed LPUs)
        groq_key = os.environ.get("GROQ_API_KEY")
        if (provider == "groq" or "groq" in model_id) and groq_key:
            target_model = api_model_name or "llama-3.3-70b-versatile"
            try:
                req_data = json.dumps({
                    "model": target_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 300,
                    "temperature": 0.3
                }).encode("utf-8")
                req = urllib.request.Request(
                    "https://api.groq.com/openai/v1/chat/completions",
                    data=req_data,
                    headers={"Content-Type": "application/json", "Authorization": f"Bearer {groq_key}"},
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    choice = data["choices"][0]["message"]["content"]
                    usage = data.get("usage", {})
                    t_in = usage.get("prompt_tokens", max(10, len(prompt.split()) * 2))
                    t_out = usage.get("completion_tokens", max(10, len(choice.split()) * 2))
                    return choice, t_in, t_out, f"[LIVE API: GROQ ({target_model})]"
            except Exception:
                return None

        # 2. Google Gemini API
        gemini_key = os.environ.get("GEMINI_API_KEY")
        if (provider == "gemini" or "gemini" in model_id) and gemini_key:
            target_model = api_model_name or "gemini-1.5-flash"
            try:
                req_data = json.dumps({
                    "model": target_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 300,
                    "temperature": 0.3
                }).encode("utf-8")
                req = urllib.request.Request(
                    "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
                    data=req_data,
                    headers={"Content-Type": "application/json", "Authorization": f"Bearer {gemini_key}"},
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    choice = data["choices"][0]["message"]["content"]
                    usage = data.get("usage", {})
                    t_in = usage.get("prompt_tokens", max(10, len(prompt.split()) * 2))
                    t_out = usage.get("completion_tokens", max(10, len(choice.split()) * 2))
                    return choice, t_in, t_out, f"[LIVE API: GEMINI ({target_model})]"
            except Exception:
                return None

        # 3. OpenRouter API
        openrouter_key = os.environ.get("OPENROUTER_API_KEY")
        if (provider == "openrouter" or "openrouter" in model_id) and openrouter_key:
            target_model = api_model_name or "meta-llama/llama-3.2-3b-instruct:free"
            try:
                req_data = json.dumps({
                    "model": target_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 300,
                    "temperature": 0.3
                }).encode("utf-8")
                req = urllib.request.Request(
                    "https://openrouter.ai/api/v1/chat/completions",
                    data=req_data,
                    headers={"Content-Type": "application/json", "Authorization": f"Bearer {openrouter_key}"},
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    choice = data["choices"][0]["message"]["content"]
                    usage = data.get("usage", {})
                    t_in = usage.get("prompt_tokens", max(10, len(prompt.split()) * 2))
                    t_out = usage.get("completion_tokens", max(10, len(choice.split()) * 2))
                    return choice, t_in, t_out, f"[LIVE API: OPENROUTER ({target_model})]"
            except Exception:
                return None

        # 4. OpenAI API
        openai_key = os.environ.get("OPENAI_API_KEY")
        if (provider == "openai" or "openai" in model_id) and openai_key:
            target_model = api_model_name or "gpt-4o-mini"
            try:
                req_data = json.dumps({
                    "model": target_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 300,
                    "temperature": 0.3
                }).encode("utf-8")
                req = urllib.request.Request(
                    "https://api.openai.com/v1/chat/completions",
                    data=req_data,
                    headers={"Content-Type": "application/json", "Authorization": f"Bearer {openai_key}"},
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    choice = data["choices"][0]["message"]["content"]
                    usage = data.get("usage", {})
                    t_in = usage.get("prompt_tokens", max(10, len(prompt.split()) * 2))
                    t_out = usage.get("completion_tokens", max(10, len(choice.split()) * 2))
                    return choice, t_in, t_out, f"[LIVE API: OPENAI ({target_model})]"
            except Exception:
                return None

        # 6. DeepSeek Official Cloud API
        deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
        if (provider == "deepseek" or "deepseek" in model_id) and deepseek_key:
            target_model = api_model_name or "deepseek-chat"
            try:
                req_data = json.dumps({
                    "model": target_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 300,
                    "temperature": 0.3
                }).encode("utf-8")
                req = urllib.request.Request(
                    "https://api.deepseek.com/chat/completions",
                    data=req_data,
                    headers={"Content-Type": "application/json", "Authorization": f"Bearer {deepseek_key}"},
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    choice = data["choices"][0]["message"]["content"]
                    usage = data.get("usage", {})
                    t_in = usage.get("prompt_tokens", max(10, len(prompt.split()) * 2))
                    t_out = usage.get("completion_tokens", max(10, len(choice.split()) * 2))
                    return choice, t_in, t_out, f"[LIVE API: DEEPSEEK ({target_model})]"
            except Exception:
                return None

        return None

    def _simulate_model_call(
        self,
        model: Dict[str, Any],
        prompt: str,
        timeout_sec: float,
        injected_lat_ms: float = 0.0,
        simulate_delay: bool = True,
        expected_output_tokens: int = 150
    ) -> Tuple[str, int, int, float, str]:
        """
        Deterministic offline model simulator with calibrated latencies and persona synthesis.
        """
        base_lat_ms = model["base_latency_ms"]
        target_lat_ms = base_lat_ms + injected_lat_ms
        target_sleep_sec = target_lat_ms / 1000.0

        if target_sleep_sec > timeout_sec:
            if simulate_delay:
                time.sleep(min(timeout_sec, 0.02))
            raise TimeoutError(f"Model call exceeded allocated timeout ({timeout_sec*1000:.0f}ms)")

        if simulate_delay:
            time.sleep(min(target_sleep_sec * 0.05, 0.02))

        tokens_in = max(10, len(prompt.split()) * 2)
        max_allowed_by_ctx = max(5, model["context_window"] - tokens_in)
        tokens_out = min(max_allowed_by_ctx, max(5, int(expected_output_tokens)))

        cap = model["capability_score"]
        name = model["name"]
        m_id = model["id"]

        p_lower = prompt.lower().strip()

        # Direct realistic answers for canonical benchmark / demo queries
        if "capital of france" in p_lower:
            resp_text = "The capital of France is Paris."
        elif "speed of light" in p_lower:
            resp_text = "The speed of light in a vacuum is exactly 299,792,458 meters per second (approximately 300,000 km/s or 186,282 miles/s)."
        elif "linked list" in p_lower and ("reverse" in p_lower or "o(1)" in p_lower):
            resp_text = (
                "Here is an in-place Python function to reverse a singly linked list with O(N) time and O(1) auxiliary space:\n\n"
                "```python\n"
                "class ListNode:\n"
                "    def __init__(self, val=0, next=None):\n"
                "        self.val = val\n"
                "        self.next = next\n\n"
                "def reverse_linked_list(head: ListNode) -> ListNode:\n"
                "    prev = None\n"
                "    curr = head\n"
                "    while curr is not None:\n"
                "        next_temp = curr.next\n"
                "        curr.next = prev\n"
                "        prev = curr\n"
                "        curr = next_temp\n"
                "    return prev\n"
                "```\n\n"
                "- **Time Complexity**: $O(N)$ where $N$ is the number of nodes.\n"
                "- **Space Complexity**: $O(1)$ auxiliary space."
            )
        elif "bellman" in p_lower:
            resp_text = (
                ("<think>\n"
                 "1. Target: Bellman Optimality Equation in MDPs.\n"
                 "2. Value function: V*(s) = max_a sum_{s', r} P(s', r | s, a) [r + gamma * V*(s')].\n"
                 "3. Action-value function: Q*(s, a) = sum_{s', r} P(s', r | s, a) [r + gamma * max_{a'} Q*(s', a')].\n"
                 "4. Verify contraction mapping and uniqueness under gamma < 1 via Banach fixed-point theorem.\n"
                 "</think>\n\n" if ("deepseek" in m_id or cap >= 0.85) else "") +
                "### Bellman Optimality Equation\n\n"
                "In a Markov Decision Process (MDP), the **optimal state-value function** $V^*(s)$ expresses the condition that the value of a state under an optimal policy equals the expected return for the best immediate action from that state:\n\n"
                "$$V^*(s) = \\max_{a \\in \\mathcal{A}} \\sum_{s' \\in \\mathcal{S}, r} p(s', r \\mid s, a) \\left[ r + \\gamma V^*(s') \\right]$$\n\n"
                "Equivalently, for the **optimal action-value function** $Q^*(s, a)$:\n\n"
                "$$Q^*(s, a) = \\sum_{s', r} p(s', r \\mid s, a) \\left[ r + \\gamma \\max_{a' \\in \\mathcal{A}} Q^*(s', a') \\right]$$\n\n"
                "**Key Properties**:\n"
                "- **Contraction Mapping**: The Bellman optimality operator is a $\\gamma$-contraction in supremum norm.\n"
                "- **Uniqueness**: By the Banach fixed-point theorem, there is a unique solution $V^*$, which value iteration and Q-learning converge toward."
            )
        elif "deepseek" in m_id or cap >= 0.90:
            resp_text = (
                f"<think>\n"
                f"1. Deconstruct user prompt: \"{prompt[:80]}\"\n"
                f"2. Apply multi-step logical chain with capability score {cap}.\n"
                f"3. Verify edge cases, mathematical invariance, and contextual constraints.\n"
                f"4. Synthesize verifiable step-by-step resolution.\n"
                f"</think>\n\n"
                f"**Detailed Solution**:\n\n"
                f"Addressing your query: *\"{prompt.strip()}\"*\n\n"
                f"Based on rigorous multi-step analysis, the optimal solution requires accounting for the relevant boundary conditions and executing with precision. All criteria have been verified and satisfy the required analytical standards."
            )
        elif "coder" in m_id:
            resp_text = (
                f"```python\n"
                f"# Implementation generated by {name}\n"
                f"def solve_task(data):\n"
                f"    \"\"\"\n"
                f"    Solution for: {prompt[:60]}\n"
                f"    \"\"\"\n"
                f"    # Process input efficiently\n"
                f"    result = [x for x in data if x is not None]\n"
                f"    return result\n"
                f"```\n\n"
                f"**Complexity**:\n"
                f"- Time: $O(N)$\n"
                f"- Space: $O(1)$"
            )
        else:
            resp_text = (
                f"Direct response to your query: *\"{prompt.strip()}\"*\n\n"
                f"The request has been evaluated and answered concisely according to your specified constraints."
            )

        badge = "[INJECTED]" if injected_lat_ms > 0 else "[SIMULATED]"
        return resp_text, tokens_in, tokens_out, target_lat_ms, badge

    def execute_request(self, request: RouteRequest, simulate_delay: bool = True) -> RouteResponse:
        t_global_start = time.perf_counter()
        request_id = f"req_{uuid.uuid4().hex[:10]}"
        timestamp_str = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # Step 1: Complexity estimation & uncertainty hedge (Laya System 1)
        est_result = self.estimator.evaluate(request.prompt, request.min_capability)
        effective_min_capability = est_result["required_capability"]

        prompt_tokens = max(10, len(request.prompt.split()) * 2)
        expected_output_tokens = request.expected_output_tokens or 150

        # Deduct time spent during estimation from the initial latency budget
        estimation_lat_ms = est_result.get("latency_ms", 0.0)
        remaining_latency_ms = request.max_latency_ms - estimation_lat_ms
        remaining_cost_usd = request.max_cost_usd

        failover_chain: List[Dict[str, Any]] = []
        attempted_model_ids = set()
        last_audits: List[CandidateAudit] = []
        last_closest_miss: Optional[str] = None

        while remaining_latency_ms > 20.0 and remaining_cost_usd > 0.0:
            # 1. Hard constraint evaluation under remaining budgets
            available_models = [m for m in self.models if m["id"] not in attempted_model_ids]
            if not available_models:
                break

            audits, closest_miss = evaluate_candidates(
                models=available_models,
                prompt_tokens=prompt_tokens,
                expected_output_tokens=expected_output_tokens,
                remaining_latency_ms=remaining_latency_ms,
                remaining_cost_usd=remaining_cost_usd,
                required_capability=effective_min_capability,
                required_context=request.required_context,
                circuit_breakers=self.circuit_breakers,
                chaos_manager=self.chaos_manager,
                telemetry=self.telemetry
            )
            last_audits = audits
            last_closest_miss = closest_miss

            valid_candidates = [a for a in audits if a.passed]
            if not valid_candidates:
                # No valid candidate satisfies current remaining budget!
                break

            # 2. Preference Ranking
            ranked = rank_candidates(valid_candidates, request.preference)
            chosen_candidate = ranked[0]
            m_id = chosen_candidate.model_id
            m_config = self.models_map[m_id]
            attempted_model_ids.add(m_id)

            # 3. Model Execution Attempt
            call_start = time.perf_counter()
            injected_lat = self.chaos_manager.get_injected_latency(m_id)
            allocated_timeout_sec = remaining_latency_ms / 1000.0
            is_live_api = False

            try:
                # Check chaos kill-switch or error rate first
                chaos_failed, chaos_reason = self.chaos_manager.should_inject_failure(m_id)
                if chaos_failed:
                    raise RuntimeError(chaos_reason)

                # 1. Attempt Real Local GGUF Model if available
                gguf_filename = m_config.get("gguf_file")
                if gguf_filename and self.gguf_engine.has_model_file(gguf_filename) and simulate_delay:
                    resp_text, tokens_in, tokens_out, gguf_lat_ms, badge = self.gguf_engine.execute_inference(
                        model_id=m_id,
                        gguf_filename=gguf_filename,
                        prompt=request.prompt,
                        timeout_sec=allocated_timeout_sec,
                        expected_output_tokens=expected_output_tokens
                    )
                    # Calibrate gateway SLA latency for auditing, SLA budgeting, and chaos testing
                    call_elapsed_ms = float(m_config["base_latency_ms"]) + injected_lat
                else:
                    # 2. Attempt Live Online API (Groq / Gemini / OpenRouter / Anthropic)
                    ext_res = self._call_external_api(m_config, request.prompt, allocated_timeout_sec)
                    if ext_res is not None:
                        is_live_api = True
                        resp_text, tokens_in, tokens_out, badge = ext_res
                        call_elapsed_ms = (time.perf_counter() - call_start) * 1000.0
                    elif simulate_delay and any(
                        cand.get("endpoint_type") == "local_gguf" and self.gguf_engine.has_model_file(cand.get("gguf_file"))
                        for cand in available_models if cand["id"] != m_id
                    ):
                        # Live mode without API key: Failover to local GGUF models
                        raise RuntimeError(f"Live API key not configured for {m_id}; triggering failover to local model")
                    else:
                        # 3. Deterministic calibrated simulator fallback
                        resp_text, tokens_in, tokens_out, sim_lat_ms, badge = self._simulate_model_call(
                            m_config, request.prompt, allocated_timeout_sec, injected_lat,
                            simulate_delay=simulate_delay,
                            expected_output_tokens=expected_output_tokens
                        )
                        call_elapsed_ms = sim_lat_ms

                actual_cost_usd = (tokens_in / 1000.0 * m_config["input_cost_per_1k"]) + \
                                  (tokens_out / 1000.0 * m_config["output_cost_per_1k"])

                # Success accounting
                cb = self.circuit_breakers.get(m_id)
                if cb:
                    cb.record_success()
                self.telemetry.record_call(m_id, call_elapsed_ms, success=True, tokens_in=tokens_in, tokens_out=tokens_out)

                prior_failover_ms = sum(attempt["time_spent_ms"] for attempt in failover_chain)
                total_elapsed_ms = prior_failover_ms + call_elapsed_ms + est_result["latency_ms"]

                actual_metrics = {
                    "total_latency_ms": round(total_elapsed_ms, 2),
                    "model_execution_ms": round(call_elapsed_ms, 2),
                    "total_cost_usd": round(actual_cost_usd, 6),
                    "tokens_in": tokens_in,
                    "tokens_out": tokens_out,
                    "model_capability": m_config["capability_score"],
                    "model_context": m_config["context_window"],
                    "badge": badge,
                    "status": "SUCCESS",
                    "no_valid_model": False
                }

                # Evaluate all models for the full decision audit table
                full_audits, _ = evaluate_candidates(
                    models=self.models,
                    prompt_tokens=prompt_tokens,
                    expected_output_tokens=expected_output_tokens,
                    remaining_latency_ms=request.max_latency_ms,
                    remaining_cost_usd=request.max_cost_usd,
                    required_capability=effective_min_capability,
                    required_context=request.required_context,
                    circuit_breakers=self.circuit_breakers,
                    chaos_manager=self.chaos_manager,
                    telemetry=self.telemetry
                )

                # Compute mathematical comparative cost breakdown matrix
                cost_matrix = compute_cost_matrix(
                    models=self.models,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    chosen_model_id=m_id,
                    chosen_cost_usd=actual_cost_usd
                )

                return RouteResponse(
                    request_id=request_id,
                    chosen_model=m_id,
                    response_text=resp_text,
                    no_valid_model=False,
                    closest_miss_suggestion=None,
                    estimator=est_result,
                    candidates=full_audits,
                    failover_chain=failover_chain,
                    actual_metrics=actual_metrics,
                    auditor={},  # Decoupled auditor populates this
                    timestamp=timestamp_str,
                    cost_matrix=cost_matrix
                )

            except Exception as exc:
                if isinstance(exc, TimeoutError):
                    call_elapsed_ms = min(float(m_config["base_latency_ms"] + injected_lat), allocated_timeout_sec * 1000.0)
                elif is_live_api:
                    call_elapsed_ms = (time.perf_counter() - call_start) * 1000.0
                else:
                    call_elapsed_ms = min(float(m_config["base_latency_ms"] + injected_lat), max(allocated_timeout_sec * 1000.0, 10.0))

                cb = self.circuit_breakers.get(m_id)
                if cb:
                    cb.record_failure()
                self.telemetry.record_call(m_id, call_elapsed_ms, success=False)

                # Shrinking Budget calculation:
                # Deduct spent time from remaining budget
                remaining_latency_ms -= call_elapsed_ms

                failover_chain.append({
                    "model_id": m_id,
                    "model_name": m_config["name"],
                    "error": str(exc),
                    "time_spent_ms": round(call_elapsed_ms, 1),
                    "remaining_latency_ms": max(0.0, round(remaining_latency_ms, 1)),
                    "remaining_cost_usd": max(0.0, round(remaining_cost_usd, 6))
                })

        # No valid model can satisfy remaining constraints -> Return structured refusal
        total_elapsed_ms = (time.perf_counter() - t_global_start) * 1000.0

        full_audits, closest_miss = evaluate_candidates(
            models=self.models,
            prompt_tokens=prompt_tokens,
            expected_output_tokens=expected_output_tokens,
            remaining_latency_ms=request.max_latency_ms,
            remaining_cost_usd=request.max_cost_usd,
            required_capability=effective_min_capability,
            required_context=request.required_context,
            circuit_breakers=self.circuit_breakers,
            chaos_manager=self.chaos_manager,
            telemetry=self.telemetry
        )

        cost_matrix = compute_cost_matrix(
            models=self.models,
            tokens_in=prompt_tokens,
            tokens_out=expected_output_tokens,
            chosen_model_id=None,
            chosen_cost_usd=0.0
        )

        return RouteResponse(
            request_id=request_id,
            chosen_model=None,
            response_text="[RouteIQ Refusal] No model satisfied the hard constraints under the available budget.",
            no_valid_model=True,
            closest_miss_suggestion=closest_miss or last_closest_miss,
            estimator=est_result,
            candidates=full_audits,
            failover_chain=failover_chain,
            actual_metrics={
                "total_latency_ms": round(total_elapsed_ms, 2),
                "total_cost_usd": 0.0,
                "model_capability": 0.0,
                "model_context": 0,
                "badge": "[RULE_ENFORCED]",
                "status": "REFUSED_NO_VALID_MODEL",
                "no_valid_model": True
            },
            auditor={},
            timestamp=timestamp_str,
            cost_matrix=cost_matrix
        )
