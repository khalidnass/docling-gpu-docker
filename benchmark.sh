#!/usr/bin/env bash
#
# benchmark.sh - Compare PDF conversion performance between the official
# docling-serve-cu128 image and the custom GPU-fixed image.
#
# Usage:
#   ./benchmark.sh
#
# Prerequisites:
#   - Docker with NVIDIA GPU support (nvidia-container-toolkit)
#   - Both images pulled/built:
#       ghcr.io/docling-project/docling-serve-cu128:latest  (official)
#       docling-serve-cu128-custom:latest                   (custom)
#
set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OFFICIAL_IMAGE="ghcr.io/docling-project/docling-serve-cu128:latest"
CUSTOM_IMAGE="docling-serve-cu128-custom:latest"
HOST_PORT=5002
API_URL="http://localhost:${HOST_PORT}/api/v2/convert/file"
HEALTH_URL="http://localhost:${HOST_PORT}/health"
RUNS=3
HEALTH_TIMEOUT=120          # seconds to wait for container to become healthy
SAMPLE_PDF="sample_attention.pdf"
SAMPLE_PDF_URL="https://arxiv.org/pdf/1706.03762"

# Container names (deterministic so we can clean up reliably)
OFFICIAL_CONTAINER="bench-docling-official"
CUSTOM_CONTAINER="bench-docling-custom"

# ---------------------------------------------------------------------------
# Arrays to store results
# ---------------------------------------------------------------------------
declare -a OFFICIAL_TIMES=()
declare -a CUSTOM_TIMES=()

# ---------------------------------------------------------------------------
# Cleanup handler - runs on EXIT regardless of success or failure
# ---------------------------------------------------------------------------
cleanup() {
    echo ""
    echo "--- Cleaning up containers ---"
    docker rm -f "${OFFICIAL_CONTAINER}" 2>/dev/null || true
    docker rm -f "${CUSTOM_CONTAINER}" 2>/dev/null || true
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Helper: wait for the service to become healthy
# ---------------------------------------------------------------------------
wait_for_healthy() {
    local label="$1"
    local elapsed=0

    echo "Waiting for ${label} to become healthy (up to ${HEALTH_TIMEOUT}s)..."
    while [ "${elapsed}" -lt "${HEALTH_TIMEOUT}" ]; do
        # Try the /health endpoint; accept any 2xx as success
        if curl -sf -o /dev/null "${HEALTH_URL}" 2>/dev/null; then
            echo "${label} is healthy after ${elapsed}s."
            return 0
        fi
        sleep 2
        elapsed=$((elapsed + 2))
    done

    echo "ERROR: ${label} did not become healthy within ${HEALTH_TIMEOUT}s."
    echo "Container logs:"
    docker logs "${2}" 2>&1 | tail -40
    return 1
}

# ---------------------------------------------------------------------------
# Helper: run a single conversion and return wall-clock seconds
# ---------------------------------------------------------------------------
run_conversion() {
    local start end duration
    start=$(date +%s%N)
    local http_code
    http_code=$(curl -s -o /dev/null -w "%{http_code}" \
        -X POST "${API_URL}" \
        -F "files=@${SAMPLE_PDF}")
    end=$(date +%s%N)

    duration=$(awk "BEGIN {printf \"%.3f\", (${end} - ${start}) / 1000000000}")

    if [ "${http_code}" -lt 200 ] || [ "${http_code}" -ge 300 ]; then
        echo "WARNING: API returned HTTP ${http_code}" >&2
    fi

    echo "${duration}"
}

# ---------------------------------------------------------------------------
# Helper: compute average of an array
# ---------------------------------------------------------------------------
compute_avg() {
    local -n arr=$1
    local sum=0
    for v in "${arr[@]}"; do
        sum=$(awk "BEGIN {printf \"%.3f\", ${sum} + ${v}}")
    done
    awk "BEGIN {printf \"%.3f\", ${sum} / ${#arr[@]}}"
}

# ---------------------------------------------------------------------------
# Step 1: Download sample PDF
# ---------------------------------------------------------------------------
echo "============================================================"
echo " Docling-Serve Benchmark: Official vs Custom Image"
echo "============================================================"
echo ""

if [ -f "${SAMPLE_PDF}" ]; then
    echo "Sample PDF already exists: ${SAMPLE_PDF}"
else
    echo "Downloading sample PDF (Attention Is All You Need)..."
    curl -L -o "${SAMPLE_PDF}" "${SAMPLE_PDF_URL}"
    echo "Downloaded: ${SAMPLE_PDF} ($(du -h "${SAMPLE_PDF}" | cut -f1))"
fi
echo ""

# ---------------------------------------------------------------------------
# Step 2: Benchmark the OFFICIAL image
# ---------------------------------------------------------------------------
echo "============================================================"
echo " Phase 1: Official Image (${OFFICIAL_IMAGE})"
echo "============================================================"
echo ""

# Remove any leftover container with the same name
docker rm -f "${OFFICIAL_CONTAINER}" 2>/dev/null || true

echo "Starting official container on port ${HOST_PORT}..."
docker run -d \
    --gpus all \
    --name "${OFFICIAL_CONTAINER}" \
    -p "${HOST_PORT}:5001" \
    "${OFFICIAL_IMAGE}"

wait_for_healthy "Official image" "${OFFICIAL_CONTAINER}"
echo ""

for i in $(seq 1 "${RUNS}"); do
    echo -n "  Run ${i}/${RUNS} ... "
    t=$(run_conversion)
    OFFICIAL_TIMES+=("${t}")
    echo "${t}s"
done

echo ""
echo "Stopping official container..."
docker rm -f "${OFFICIAL_CONTAINER}" >/dev/null 2>&1
echo ""

# ---------------------------------------------------------------------------
# Step 3: Benchmark the CUSTOM image
# ---------------------------------------------------------------------------
echo "============================================================"
echo " Phase 2: Custom Image (${CUSTOM_IMAGE})"
echo "============================================================"
echo ""

# Remove any leftover container with the same name
docker rm -f "${CUSTOM_CONTAINER}" 2>/dev/null || true

echo "Starting custom container on port ${HOST_PORT}..."
docker run -d \
    --gpus all \
    --name "${CUSTOM_CONTAINER}" \
    -p "${HOST_PORT}:5001" \
    "${CUSTOM_IMAGE}"

wait_for_healthy "Custom image" "${CUSTOM_CONTAINER}"
echo ""

for i in $(seq 1 "${RUNS}"); do
    echo -n "  Run ${i}/${RUNS} ... "
    t=$(run_conversion)
    CUSTOM_TIMES+=("${t}")
    echo "${t}s"
done

echo ""
echo "Stopping custom container..."
docker rm -f "${CUSTOM_CONTAINER}" >/dev/null 2>&1
echo ""

# ---------------------------------------------------------------------------
# Step 4: Print comparison table
# ---------------------------------------------------------------------------
OFFICIAL_AVG=$(compute_avg OFFICIAL_TIMES)
CUSTOM_AVG=$(compute_avg CUSTOM_TIMES)

# Compute speedup (official / custom); >1 means custom is faster
SPEEDUP=$(awk "BEGIN {
    if (${CUSTOM_AVG} > 0)
        printf \"%.2f\", ${OFFICIAL_AVG} / ${CUSTOM_AVG};
    else
        printf \"N/A\"
}")

echo "============================================================"
echo " RESULTS"
echo "============================================================"
echo ""
printf "%-12s | %-14s | %-14s | %-14s | %-10s\n" \
    "Image" "Run 1 (s)" "Run 2 (s)" "Run 3 (s)" "Avg (s)"
printf "%-12s-+-%-14s-+-%-14s-+-%-14s-+-%-10s\n" \
    "------------" "--------------" "--------------" "--------------" "----------"
printf "%-12s | %-14s | %-14s | %-14s | %-10s\n" \
    "Official" "${OFFICIAL_TIMES[0]}" "${OFFICIAL_TIMES[1]}" "${OFFICIAL_TIMES[2]}" "${OFFICIAL_AVG}"
printf "%-12s | %-14s | %-14s | %-14s | %-10s\n" \
    "Custom" "${CUSTOM_TIMES[0]}" "${CUSTOM_TIMES[1]}" "${CUSTOM_TIMES[2]}" "${CUSTOM_AVG}"
echo ""
echo "Speedup (Official avg / Custom avg): ${SPEEDUP}x"
if [ "$(awk "BEGIN {print (${SPEEDUP} > 1.0) ? 1 : 0}")" -eq 1 ]; then
    echo "  -> Custom image is ${SPEEDUP}x faster than official."
elif [ "$(awk "BEGIN {print (${SPEEDUP} < 1.0) ? 1 : 0}")" -eq 1 ]; then
    INVERSE=$(awk "BEGIN {printf \"%.2f\", 1.0 / ${SPEEDUP}}")
    echo "  -> Official image is ${INVERSE}x faster than custom."
else
    echo "  -> Both images performed equally."
fi
echo ""
echo "Done."
