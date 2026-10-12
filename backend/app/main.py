import json
import time
import os
import sys
import datetime
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from typing import Dict, Any, Optional

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

try:
    from .config import MODELS_REGISTRY_PATH, DB_PATH, BENCHMARK_PATH
    from .models.schemas import RouteRequest, RouteResponse
    from .estimators.multi_estimator import MultiEstimatorConsensus
    from .engine.circuit_breaker import CircuitBreaker
    from .chaos.manager import ChaosManager
    from .storage.telemetry import TelemetryStore
    from .storage.db import AuditDatabase, GENESIS_HASH
    from .engine.auditor import IndependentAuditor
    from .engine.executor import ResilientExecutor
except (ImportError, ValueError):
    from backend.app.config import MODELS_REGISTRY_PATH, DB_PATH, BENCHMARK_PATH
    from backend.app.models.schemas import RouteRequest, RouteResponse
    from backend.app.estimators.multi_estimator import MultiEstimatorConsensus
    from backend.app.engine.circuit_breaker import CircuitBreaker
    from backend.app.chaos.manager import ChaosManager
    from backend.app.storage.telemetry import TelemetryStore
    from backend.app.storage.db import AuditDatabase, GENESIS_HASH
    from backend.app.engine.auditor import IndependentAuditor
    from backend.app.engine.executor import ResilientExecutor

# Initialize Core Singletons
with open(MODELS_REGISTRY_PATH, "r", encoding="utf-8") as f:
    MODELS_REGISTRY = json.load(f)

CIRCUIT_BREAKERS = {m["id"]: CircuitBreaker() for m in MODELS_REGISTRY}
CHAOS_MANAGER = ChaosManager()
TELEMETRY_STORE = TelemetryStore()
AUDIT_DB = AuditDatabase()
ESTIMATOR_CONSENSUS = MultiEstimatorConsensus()

EXECUTOR = ResilientExecutor(
    models=MODELS_REGISTRY,
    circuit_breakers=CIRCUIT_BREAKERS,
    chaos_manager=CHAOS_MANAGER,
    telemetry=TELEMETRY_STORE,
    estimator=ESTIMATOR_CONSENSUS
)

def handle_route_request(req_data: Dict[str, Any]) -> Dict[str, Any]:
    api_keys = req_data.get("api_keys")
    if isinstance(api_keys, dict):
        for k, v in api_keys.items():
            if v and isinstance(v, str) and v.strip():
                os.environ[k.strip()] = v.strip()

    prompt = req_data.get("prompt", "")
    max_lat = float(req_data.get("max_latency_ms") if req_data.get("max_latency_ms") is not None else 2000.0)
    max_cost = float(req_data.get("max_cost_usd") if req_data.get("max_cost_usd") is not None else 0.02)
    min_cap = float(req_data.get("min_capability") if req_data.get("min_capability") is not None else 0.0)
    req_ctx = req_data.get("required_context")
    if req_ctx is not None:
        req_ctx = int(req_ctx)
    pref = req_data.get("preference", "lowest_cost")
    exp_tokens = int(req_data.get("expected_output_tokens") if req_data.get("expected_output_tokens") is not None else 150)

    route_req = RouteRequest(
        prompt=prompt,
        max_latency_ms=max_lat,
        max_cost_usd=max_cost,
        min_capability=min_cap,
        required_context=req_ctx,
        preference=pref,
        expected_output_tokens=exp_tokens
    )

    # 1. Execute routing
    response = EXECUTOR.execute_request(route_req)

    # 2. Independent post-hoc audit
    req_dict = route_req.model_dump() if hasattr(route_req, "model_dump") else route_req.dict()
    resp_dict = response.model_dump() if hasattr(response, "model_dump") else response.dict()

    actuals_for_audit = dict(resp_dict.get("actual_metrics", {}))
    actuals_for_audit["no_valid_model"] = resp_dict.get("no_valid_model", False)

    audit_verdict = IndependentAuditor.audit_record(req_dict, actuals_for_audit)
    response.auditor = audit_verdict
    resp_dict["auditor"] = audit_verdict

    # 3. Cryptographic hash-chained SQLite record
    curr_hash = AUDIT_DB.insert_audit_record(req_dict, resp_dict, audit_verdict)
    resp_dict["auditor"]["chain_hash"] = curr_hash

    return resp_dict

def run_workload_comparison(num_requests: int = 30) -> Dict[str, Any]:
    """
    Runs side-by-side benchmark comparison:
    1. Always Strongest (Claude 3.5 / DeepSeek 14B)
    2. Always Cheapest (Llama 3.2 1B)
    3. RouteIQ (Dynamic Constraint Routing)
    """
    bench_data = []
    if BENCHMARK_PATH.exists():
        with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
            bench_data = json.load(f)

    if not bench_data:
        bench_data = [
            {"prompt": f"Benchmark task {i}: write an algorithm and optimize memory.", "expected_tier": "medium", "min_capability": 0.65}
            for i in range(num_requests)
        ]

    tasks = bench_data[:num_requests]

    # Model specs
    strongest_model = max(MODELS_REGISTRY, key=lambda m: m.get("capability_score", 0.0))
    cheapest_model = next((m for m in MODELS_REGISTRY if m["id"] == "llama-3.2-1b"), MODELS_REGISTRY[0])

    # Simulation accumulators
    results = {
        "always_strongest": {"cost": 0.0, "latencies": [], "violations": 0, "success": 0},
        "always_cheapest": {"cost": 0.0, "latencies": [], "violations": 0, "success": 0},
        "routeiq": {"cost": 0.0, "latencies": [], "violations": 0, "success": 0}
    }

    for task in tasks:
        prompt = task["prompt"]
        tokens_in = max(10, len(prompt.split()) * 2)
        tokens_out = 120

        # Constraints
        max_lat = 1000.0
        max_cost = 0.005
        min_cap = task.get("min_capability", 0.50)

        # 1. Always Strongest
        c_cost = (tokens_in / 1000.0 * strongest_model["input_cost_per_1k"]) + (tokens_out / 1000.0 * strongest_model["output_cost_per_1k"])
        c_lat = strongest_model["base_latency_ms"]
        results["always_strongest"]["cost"] += c_cost
        results["always_strongest"]["latencies"].append(c_lat)
        # Latency violation check
        if c_lat > max_lat or c_cost > max_cost:
            results["always_strongest"]["violations"] += 1
        results["always_strongest"]["success"] += 1

        # 2. Always Cheapest
        l_cost = (tokens_in / 1000.0 * cheapest_model["input_cost_per_1k"]) + (tokens_out / 1000.0 * cheapest_model["output_cost_per_1k"])
        l_lat = cheapest_model["base_latency_ms"]
        results["always_cheapest"]["cost"] += l_cost
        results["always_cheapest"]["latencies"].append(l_lat)
        # Capability violation check
        if cheapest_model["capability_score"] < min_cap:
            results["always_cheapest"]["violations"] += 1
        results["always_cheapest"]["success"] += 1

        # 3. RouteIQ
        r_resp = EXECUTOR.execute_request(RouteRequest(
            prompt=prompt,
            max_latency_ms=max_lat,
            max_cost_usd=max_cost,
            min_capability=min_cap,
            preference="lowest_cost",
            expected_output_tokens=tokens_out
        ), simulate_delay=False)
        actuals = r_resp.actual_metrics
        results["routeiq"]["cost"] += actuals.get("total_cost_usd", 0.0)
        results["routeiq"]["latencies"].append(actuals.get("total_latency_ms", 0.0))
        # RouteIQ never violates constraints
        if not r_resp.no_valid_model and (actuals.get("total_latency_ms", 0.0) > max_lat + 5.0 or actuals.get("total_cost_usd", 0.0) > max_cost * 1.01):
            results["routeiq"]["violations"] += 1
        results["routeiq"]["success"] += 1

    # Aggregate summaries
    def summarize(k):
        lats = sorted(results[k]["latencies"])
        p50 = lats[len(lats)//2] if lats else 0.0
        p95 = lats[int(len(lats)*0.95)] if lats else 0.0
        return {
            "total_cost_usd": round(results[k]["cost"], 6),
            "avg_latency_ms": round(sum(lats) / len(lats), 1) if lats else 0.0,
            "p50_latency_ms": round(p50, 1),
            "p95_latency_ms": round(p95, 1),
            "violations": results[k]["violations"],
            "violation_rate_pct": round((results[k]["violations"] / num_requests) * 100, 1),
            "success_count": results[k]["success"]
        }

    strongest_sum = summarize("always_strongest")
    cheapest_sum = summarize("always_cheapest")
    routeiq_sum = summarize("routeiq")

    savings_vs_strongest = (
        ((strongest_sum["total_cost_usd"] - routeiq_sum["total_cost_usd"]) / strongest_sum["total_cost_usd"] * 100.0)
        if strongest_sum["total_cost_usd"] > 0 else 0.0
    )

    return {
        "num_requests": num_requests,
        "cost_savings_vs_strongest_pct": round(savings_vs_strongest, 1),
        "always_strongest": strongest_sum,
        "always_cheapest": cheapest_sum,
        "routeiq": routeiq_sum
    }


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True

class RouteIQHandler(BaseHTTPRequestHandler):
    def _send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def do_OPTIONS(self):
        self.send_response(204)
        self._send_cors_headers()
        self.end_headers()

    def _send_json(self, status: int, data: Any):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]

        if path == "/health":
            violations, total = AUDIT_DB.get_violation_counts()
            chain_status = AUDIT_DB.verify_chain_integrity()
            self._send_json(200, {
                "status": "ok",
                "service": "RouteIQ AI Gateway",
                "version": "1.0.0",
                "models_registered": len(MODELS_REGISTRY),
                "auditor_violations": violations,
                "total_audited_requests": total,
                "chain_valid": chain_status.get("valid", True),
                "genesis_hash": GENESIS_HASH
            })

        elif path == "/models":
            models_list = []
            gguf_cat = EXECUTOR.gguf_engine.get_catalog()
            for m in MODELS_REGISTRY:
                m_id = m["id"]
                cb = CIRCUIT_BREAKERS.get(m_id)
                ch = CHAOS_MANAGER.get_chaos(m_id)
                stats = TELEMETRY_STORE.get_stats(m_id)
                g_file = m.get("gguf_file")
                g_meta = gguf_cat.get(g_file, {}) if g_file else {}
                models_list.append({
                    **m,
                    "circuit_breaker": cb.get_info() if cb else {},
                    "chaos": ch,
                    "telemetry": stats,
                    "gguf_meta": g_meta
                })
            self._send_json(200, models_list)

        elif path == "/chaos":
            self._send_json(200, CHAOS_MANAGER.get_all_chaos())

        elif path == "/audit":
            records = AUDIT_DB.get_audit_records(limit=100)
            violations, total = AUDIT_DB.get_violation_counts()
            chain_status = AUDIT_DB.verify_chain_integrity()
            self._send_json(200, {
                "total_records": total,
                "violations_count": violations,
                "chain_valid": chain_status["valid"],
                "chain_status": chain_status,
                "records": records
            })

        elif path == "/audit/verify":
            chain_status = AUDIT_DB.verify_chain_integrity()
            self._send_json(200, chain_status)

        elif path == "/telemetry":
            self._send_json(200, TELEMETRY_STORE.get_all_stats())

        elif path == "/keys":
            self._send_json(200, {
                "GROQ_API_KEY": bool(os.environ.get("GROQ_API_KEY")),
                "GEMINI_API_KEY": bool(os.environ.get("GEMINI_API_KEY")),
                "OPENROUTER_API_KEY": bool(os.environ.get("OPENROUTER_API_KEY")),
                "OPENAI_API_KEY": bool(os.environ.get("OPENAI_API_KEY")),
                "DEEPSEEK_API_KEY": bool(os.environ.get("DEEPSEEK_API_KEY")),
                "ANTHROPIC_API_KEY": bool(os.environ.get("ANTHROPIC_API_KEY")),
            })

        elif path == "/events":
            # Server-Sent Events endpoint
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self._send_cors_headers()
            self.end_headers()

            try:
                for _ in range(10):
                    data = json.dumps({
                        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        "telemetry": TELEMETRY_STORE.get_all_stats(),
                        "violations": AUDIT_DB.get_violation_counts()[0]
                    })
                    self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
                    self.wfile.flush()
                    time.sleep(1.0)
            except (BrokenPipeError, ConnectionResetError):
                pass

        elif path == "/" or path.startswith("/static"):
            # Static file serving if frontend dist exists
            dist_dir = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"
            file_path = dist_dir / (path[1:] if path != "/" else "index.html")
            if file_path.exists() and file_path.is_file():
                content_type = "text/html"
                if file_path.suffix == ".js":
                    content_type = "application/javascript"
                elif file_path.suffix == ".css":
                    content_type = "text/css"
                elif file_path.suffix == ".json":
                    content_type = "application/json"
                elif file_path.suffix == ".svg":
                    content_type = "image/svg+xml"
                elif file_path.suffix in (".jpg", ".jpeg"):
                    content_type = "image/jpeg"
                elif file_path.suffix == ".png":
                    content_type = "image/png"

                with open(file_path, "rb") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(content)))
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(content)
            else:
                self._send_json(200, {
                    "service": "RouteIQ AI Gateway",
                    "status": "online",
                    "docs": "Access /models, /health, /audit, /chaos, or POST /route"
                })
        else:
            self._send_json(404, {"error": "Not Found", "path": path})

    def do_POST(self):
        path = self.path.split("?")[0]
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"
        try:
            req_data = json.loads(body)
        except Exception:
            self._send_json(400, {"error": "Invalid JSON body"})
            return

        if path == "/route":
            try:
                resp = handle_route_request(req_data)
                self._send_json(200, resp)
            except ValueError as ve:
                self._send_json(400, {"error": str(ve)})
            except Exception as e:
                self._send_json(500, {"error": str(e)})

        elif path == "/chaos":
            model_id = req_data.get("model_id")
            if not model_id:
                self._send_json(400, {"error": "model_id required"})
                return
            CHAOS_MANAGER.set_chaos(
                model_id=model_id,
                add_latency_ms=req_data.get("add_latency_ms", 0.0),
                kill_switch=req_data.get("kill_switch", False),
                error_rate=req_data.get("error_rate", 0.0)
            )
            self._send_json(200, {"status": "updated", "chaos": CHAOS_MANAGER.get_chaos(model_id)})

        elif path == "/chaos/reset":
            CHAOS_MANAGER.reset_all()
            for cb in CIRCUIT_BREAKERS.values():
                cb.reset()
            TELEMETRY_STORE.reset()
            self._send_json(200, {"status": "all chaos, circuit breakers, and telemetry reset"})

        elif path == "/audit/clear":
            AUDIT_DB.clear()
            self._send_json(200, {"status": "audit database cleared"})

        elif path == "/benchmark/compare":
            count = int(req_data.get("count", 30))
            comp_res = run_workload_comparison(num_requests=count)
            self._send_json(200, comp_res)

        elif path == "/keys":
            for k in ["GROQ_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY"]:
                if k in req_data:
                    v = str(req_data[k]).strip()
                    if v:
                        os.environ[k] = v
            try:
                env_p = Path(__file__).resolve().parent.parent.parent / ".env"
                lines = []
                for k in ["GROQ_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY"]:
                    if os.environ.get(k):
                        lines.append(f"{k}={os.environ.get(k)}\n")
                with open(env_p, "w", encoding="utf-8") as f:
                    f.writelines(lines)
            except Exception:
                pass
            self._send_json(200, {
                "status": "updated",
                "keys": {
                    "GROQ_API_KEY": bool(os.environ.get("GROQ_API_KEY")),
                    "GEMINI_API_KEY": bool(os.environ.get("GEMINI_API_KEY")),
                    "OPENROUTER_API_KEY": bool(os.environ.get("OPENROUTER_API_KEY")),
                    "OPENAI_API_KEY": bool(os.environ.get("OPENAI_API_KEY")),
                    "DEEPSEEK_API_KEY": bool(os.environ.get("DEEPSEEK_API_KEY")),
                }
            })

        else:
            self._send_json(404, {"error": "Not Found", "path": path})

    def log_message(self, format, *args):
        # Concise logging
        sys.stderr.write(f"[{self.log_date_time_string()}] {self.address_string()} {format % args}\n")

try:
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, HTMLResponse
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(title="RouteIQ Gateway", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/route")
    async def api_route(req: Dict[str, Any]):
        return handle_route_request(req)

    @app.get("/health")
    async def api_health():
        violations, total = AUDIT_DB.get_violation_counts()
        chain_status = AUDIT_DB.verify_chain_integrity()
        return {
            "status": "ok",
            "service": "RouteIQ AI Gateway",
            "version": "1.0.0",
            "models_registered": len(MODELS_REGISTRY),
            "auditor_violations": violations,
            "total_audited_requests": total,
            "chain_valid": chain_status.get("valid", True),
            "genesis_hash": GENESIS_HASH
        }

    @app.get("/models")
    async def api_models():
        models_list = []
        gguf_cat = EXECUTOR.gguf_engine.get_catalog()
        for m in MODELS_REGISTRY:
            m_id = m["id"]
            cb = CIRCUIT_BREAKERS.get(m_id)
            ch = CHAOS_MANAGER.get_chaos(m_id)
            stats = TELEMETRY_STORE.get_stats(m_id)
            g_file = m.get("gguf_file")
            g_meta = gguf_cat.get(g_file, {}) if g_file else {}
            models_list.append({
                **m,
                "circuit_breaker": cb.get_info() if cb else {},
                "chaos": ch,
                "telemetry": stats,
                "gguf_meta": g_meta
            })
        return models_list

    @app.post("/chaos")
    async def api_chaos(data: Dict[str, Any]):
        model_id = data.get("model_id")
        CHAOS_MANAGER.set_chaos(
            model_id=model_id,
            add_latency_ms=data.get("add_latency_ms", 0.0),
            kill_switch=data.get("kill_switch", False),
            error_rate=data.get("error_rate", 0.0)
        )
        return {"status": "updated", "chaos": CHAOS_MANAGER.get_chaos(model_id)}

    @app.post("/chaos/reset")
    async def api_chaos_reset():
        CHAOS_MANAGER.reset_all()
        for cb in CIRCUIT_BREAKERS.values():
            cb.reset()
        return {"status": "all chaos and circuit breakers reset"}

    @app.get("/audit")
    async def api_audit():
        records = AUDIT_DB.get_audit_records(limit=100)
        violations, total = AUDIT_DB.get_violation_counts()
        chain_status = AUDIT_DB.verify_chain_integrity()
        return {
            "total_records": total,
            "violations_count": violations,
            "chain_valid": chain_status["valid"],
            "chain_status": chain_status,
            "records": records
        }

    @app.get("/audit/verify")
    async def api_audit_verify():
        return AUDIT_DB.verify_chain_integrity()

    @app.post("/audit/clear")
    async def api_audit_clear():
        AUDIT_DB.clear()
        return {"status": "audit database cleared"}

    @app.post("/benchmark/compare")
    async def api_benchmark(data: Dict[str, Any] = None):
        count = int((data or {}).get("count", 30))
        return run_workload_comparison(num_requests=count)

    @app.get("/telemetry")
    async def api_telemetry():
        return TELEMETRY_STORE.get_all_stats()

    @app.get("/keys")
    async def api_get_keys():
        return {
            "GROQ_API_KEY": bool(os.environ.get("GROQ_API_KEY")),
            "GEMINI_API_KEY": bool(os.environ.get("GEMINI_API_KEY")),
            "OPENROUTER_API_KEY": bool(os.environ.get("OPENROUTER_API_KEY")),
            "OPENAI_API_KEY": bool(os.environ.get("OPENAI_API_KEY")),
            "DEEPSEEK_API_KEY": bool(os.environ.get("DEEPSEEK_API_KEY")),
        }

    @app.post("/keys")
    async def api_post_keys(req_data: Dict[str, Any]):
        for k in ["GROQ_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY"]:
            if k in req_data:
                v = str(req_data[k]).strip()
                if v:
                    os.environ[k] = v
        try:
            env_p = Path(__file__).resolve().parent.parent.parent / ".env"
            lines = []
            for k in ["GROQ_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY"]:
                if os.environ.get(k):
                    lines.append(f"{k}={os.environ.get(k)}\n")
            with open(env_p, "w", encoding="utf-8") as f:
                f.writelines(lines)
        except Exception:
            pass
        return {
            "status": "updated",
            "keys": {
                "GROQ_API_KEY": bool(os.environ.get("GROQ_API_KEY")),
                "GEMINI_API_KEY": bool(os.environ.get("GEMINI_API_KEY")),
                "OPENROUTER_API_KEY": bool(os.environ.get("OPENROUTER_API_KEY")),
                "OPENAI_API_KEY": bool(os.environ.get("OPENAI_API_KEY")),
                "DEEPSEEK_API_KEY": bool(os.environ.get("DEEPSEEK_API_KEY")),
            }
        }
    dist_dir = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"
    index_file = dist_dir / "index.html"

    @app.get("/", response_class=HTMLResponse)
    async def api_root():
        if index_file.exists():
            with open(index_file, "r", encoding="utf-8") as f:
                return f.read()
        return "<html><body><h1>RouteIQ AI Gateway</h1><p>Online. Dashboard assets missing.</p></body></html>"

    @app.get("/landing_bg.jpg")
    async def api_landing_bg():
        bg_file = dist_dir / "landing_bg.jpg"
        if bg_file.exists():
            from fastapi.responses import FileResponse
            return FileResponse(bg_file, media_type="image/jpeg")
        from fastapi import Response
        return Response(status_code=404)
except ImportError:
    app = None

def run_server(host: str = "0.0.0.0", port: int = 8000):
    try:
        import uvicorn
        print(f"[RouteIQ] Gateway Server running via Uvicorn/FastAPI at http://{host}:{port}")
        uvicorn.run(app, host=host, port=port, log_level="info")
    except Exception as exc:
        print(f"[RouteIQ] Uvicorn unavailable ({exc}), falling back to ThreadedHTTPServer at http://{host}:{port}")
        server = ThreadedHTTPServer((host, port), RouteIQHandler)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nStopping RouteIQ Gateway Server...")
            server.server_close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="RouteIQ Gateway")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)), help="Port number")
    parser.add_argument("--host", type=str, default=os.environ.get("HOST", "0.0.0.0"), help="Host address")
    args, _ = parser.parse_known_args()
    run_server(host=args.host, port=args.port)
