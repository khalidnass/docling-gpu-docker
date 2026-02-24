#!/usr/bin/env bash
set -e

echo "============================================================"
echo "  DOCLING-SERVE DEBUG IMAGE"
echo "  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "  ENABLE_PROFILING=${ENABLE_PROFILING:-true}"
echo "============================================================"

# ---- Ensure cache dirs are writable for UID 1001 ----
(chown -R 1001:0 /opt/app-root/src/.cache 2>/dev/null || true)
(chmod -R g+rwX /opt/app-root/src/.cache 2>/dev/null || true)

# ---- Fix DocumentFigureClassifier v2.0 for volume-mounted models ----
MODELS="${DOCLING_SERVE_ARTIFACTS_PATH:-/opt/app-root/src/.cache/docling/models}"
BAKED=/opt/app-root/docling-project--DocumentFigureClassifier-v2.0-baked
TARGET="$MODELS/docling-project--DocumentFigureClassifier-v2.0"

if [ ! -d "$TARGET" ] || [ ! -f "$TARGET/config.json" ]; then
    echo "[entrypoint] Copying baked-in DocumentFigureClassifier-v2.0 to $TARGET"
    rm -rf "$TARGET" 2>/dev/null || true
    cp -a "$BAKED" "$TARGET" 2>/dev/null || {
        echo "[entrypoint] WARNING: Could not copy classifier model. Trying symlink..."
        ln -sf "$BAKED" "$TARGET" 2>/dev/null || echo "[entrypoint] WARNING: Classification may not work"
    }
fi

if [ ! -e "$MODELS/docling-project--DocumentFigureClassifier" ]; then
    ln -sf "$TARGET" "$MODELS/docling-project--DocumentFigureClassifier" 2>/dev/null || true
fi

# ---- Debug mode: apply profiling patches and print diagnostics ----
DEBUG_VAL="${ENABLE_PROFILING:-true}"
if [ "$DEBUG_VAL" = "true" ] || [ "$DEBUG_VAL" = "1" ] || [ "$DEBUG_VAL" = "yes" ]; then
    echo "[entrypoint] Debug mode ENABLED — [PROFILE] output active"
    echo "[entrypoint] Set ENABLE_PROFILING=false to disable profiling"
    echo "[entrypoint] Installing debug profiling patches..."
    python3 -c "
from docling.docling_debug_profiler import install, print_startup_diagnostics
install()
print_startup_diagnostics()
" 2>&1 || echo "[entrypoint] WARNING: Profiling patches failed to install"
else
    echo "[entrypoint] Debug mode DISABLED — no [PROFILE] output"
fi

# ---- Run the original command ----
echo "[entrypoint] Starting: $@"
echo "============================================================"
exec "$@"
