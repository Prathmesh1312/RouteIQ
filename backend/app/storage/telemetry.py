import threading
import time
from typing import Dict, List, Optional, Any

class TelemetryStore:
    """
    In-memory rolling telemetry store.
    Tracks rolling EMA latency, p95 latency, request counts, and error rates per model.
    """
    def __init__(self, alpha: float = 0.2, max_samples: int = 50):
        self._lock = threading.RLock()
        self.alpha = alpha  # EMA smoothing factor
        self.max_samples = max_samples
        self.ema_latencies: Dict[str, float] = {}
        self.recent_latencies: Dict[str, List[float]] = {}
        self.request_counts: Dict[str, int] = {}
        self.error_counts: Dict[str, int] = {}
        self.total_tokens_in: Dict[str, int] = {}
        self.total_tokens_out: Dict[str, int] = {}
    def reset(self):
        with self._lock:
            self.ema_latencies.clear()
            self.recent_latencies.clear()
            self.request_counts.clear()
            self.error_counts.clear()
            self.total_tokens_in.clear()
            self.total_tokens_out.clear()

    def record_call(self, model_id: str, latency_ms: float, success: bool, tokens_in: int = 0, tokens_out: int = 0):
        with self._lock:
            # Request count
            self.request_counts[model_id] = self.request_counts.get(model_id, 0) + 1
            if not success:
                self.error_counts[model_id] = self.error_counts.get(model_id, 0) + 1

            self.total_tokens_in[model_id] = self.total_tokens_in.get(model_id, 0) + tokens_in
            self.total_tokens_out[model_id] = self.total_tokens_out.get(model_id, 0) + tokens_out

            # EMA
            if model_id not in self.ema_latencies:
                self.ema_latencies[model_id] = latency_ms
            else:
                self.ema_latencies[model_id] = (self.alpha * latency_ms) + ((1 - self.alpha) * self.ema_latencies[model_id])

            # Recent samples for p95
            samples = self.recent_latencies.setdefault(model_id, [])
            samples.append(latency_ms)
            if len(samples) > self.max_samples:
                samples.pop(0)

    def get_rolling_p95_latency(self, model_id: str) -> Optional[float]:
        with self._lock:
            samples = self.recent_latencies.get(model_id, [])
            if not samples:
                return None
            sorted_samples = sorted(samples)
            idx = int(len(sorted_samples) * 0.95)
            idx = min(idx, len(sorted_samples) - 1)
            return sorted_samples[idx]

    def get_ema_latency(self, model_id: str) -> Optional[float]:
        with self._lock:
            return self.ema_latencies.get(model_id)

    def get_stats(self, model_id: str) -> Dict[str, Any]:
        with self._lock:
            reqs = self.request_counts.get(model_id, 0)
            errs = self.error_counts.get(model_id, 0)
            error_rate = (errs / reqs) if reqs > 0 else 0.0
            return {
                "requests": reqs,
                "errors": errs,
                "error_rate": round(error_rate, 4),
                "ema_latency_ms": round(self.ema_latencies.get(model_id, 0.0), 1),
                "p95_latency_ms": round(self.get_rolling_p95_latency(model_id) or 0.0, 1),
                "total_tokens_in": self.total_tokens_in.get(model_id, 0),
                "total_tokens_out": self.total_tokens_out.get(model_id, 0)
            }

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            all_ids = set(self.request_counts.keys()) | set(self.ema_latencies.keys())
        return {m_id: self.get_stats(m_id) for m_id in all_ids}
