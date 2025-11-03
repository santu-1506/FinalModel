#!/bin/bash
# Entrypoint script for CRISPR Model API

set -e

# Get port from environment variable or default to 8080
PORT="${PORT:-8080}"

echo "=========================================="
echo "Starting CRISPR Model API on port $PORT"
echo "=========================================="

# Start gunicorn with the Flask app
exec gunicorn \
    --bind 0.0.0.0:$PORT \
    --workers 1 \
    --threads 2 \
    --timeout 600 \
    --access-logfile - \
    --error-logfile - \
    --log-level info \
    --preload \
    model_api_final:app

