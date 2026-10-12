#!/usr/bin/env bash
set -e

# ==============================================================================
# RouteIQ: Intelligent Cost-, Latency- & Quality-Aware AI Gateway
# VORTEXA 3.0 — Domain 2: AI Systems & Infrastructure (Problem Statement A-01)
# One-Command Bootstrapper & Verification Runner
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=================================================================="
echo "⚡ RouteIQ: Dynamic Cost-, Latency- & Quality-Aware AI Gateway"
echo "   Domain 2: AI Systems & Infrastructure (VORTEXA 3.0)"
echo "=================================================================="

# 1. Environment Detection: Prefer python environment with sklearn
if command -v python3 >/dev/null 2>&1 && python3 -c "import sklearn" >/dev/null 2>&1; then
    PYTHON_BIN="python3"
elif [ -d "../venv/bin" ]; then
    PYTHON_BIN="../venv/bin/python"
elif [ -d "venv/bin" ]; then
    PYTHON_BIN="venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
else
    echo "❌ Error: Python 3 not found!"
    exit 1
fi

export PYTHONPATH=".:$PYTHONPATH"
export ENABLE_LLAMA_CPP="${ENABLE_LLAMA_CPP:-1}"

# 2. Command Line Arguments Dispatcher
case "$1" in
    --test)
        echo "🧪 Running RouteIQ Test Suite (T1-T12 Acceptance + GGUF Integration + 1,000 Invariants)..."
        echo "------------------------------------------------------------------"
        $PYTHON_BIN -m unittest backend/tests/test_acceptance.py -v
        echo ""
        $PYTHON_BIN -m unittest backend/tests/test_gguf_integration.py -v
        echo ""
        $PYTHON_BIN -m unittest backend/tests/test_property_1000.py -v
        echo "------------------------------------------------------------------"
        echo "✅ All tests passed! Independent Auditor verified 0 violations."
        exit 0
        ;;

    --benchmark)
        echo "📊 Running Side-by-Side Workload Benchmark Comparison..."
        $PYTHON_BIN backend/scripts/run_comparison.py
        exit 0
        ;;

    --eval)
        echo "🎯 Running Model Capability & Benchmark Task Evaluation..."
        $PYTHON_BIN backend/scripts/run_benchmark.py
        exit 0
        ;;

    --train)
        echo "🤖 Training & Calibrating Estimator Model..."
        $PYTHON_BIN backend/scripts/train_estimator.py
        exit 0
        ;;
esac

# 3. Model & Estimator Verification
if [ ! -f "backend/data/estimator_tfidf.joblib" ]; then
    echo "📦 Calibrating baseline estimator..."
    $PYTHON_BIN backend/scripts/train_estimator.py >/dev/null 2>&1 || true
fi

# 4. Port Configuration
PORT=${PORT:-8000}

echo ""
echo "🚀 Starting RouteIQ Gateway Server on port $PORT..."
echo "------------------------------------------------------------------"
echo "🌐 Interactive Judge UI Dashboard: http://localhost:$PORT"
echo "📡 REST API Endpoints:"
echo "   • POST http://localhost:$PORT/route              (Inference Gateway)"
echo "   • GET  http://localhost:$PORT/health             (Health & Audit Status)"
echo "   • GET  http://localhost:$PORT/models             (Registered Models & Telemetry)"
echo "   • POST http://localhost:$PORT/chaos              (Judge Chaos Studio)"
echo "   • GET  http://localhost:$PORT/audit              (SHA-256 Chained Logs)"
echo "   • GET  http://localhost:$PORT/audit/verify       (Cryptographic Chain Proof)"
echo "   • POST http://localhost:$PORT/benchmark/compare  (Comparative ROI Benchmark)"
echo "------------------------------------------------------------------"
echo "Core Axiom: Rules decide. AI only advises. Failover can never relax a constraint."
echo "Press Ctrl+C to stop."
echo ""

exec $PYTHON_BIN -m backend.app.main
