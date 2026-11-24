#!/usr/bin/env bash
# Measure embedding latency inside the gateway container (its CPU and memory limits apply).
# Extra arguments go to measure_embed_latency.py, e.g. --threads 1 2 4 --concurrency 1 16.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ "$(uname)" == "Darwin" ]]; then
  host_cpu="$(sysctl -n machdep.cpu.brand_string)"
  host_mem="$(sysctl -n hw.memsize)"
  host_os="macOS $(sw_vers -productVersion)"
else
  host_cpu="$(lscpu | sed -n 's/^Model name: *//p' | head -1)"
  host_mem="$(awk '/MemTotal/ {print $2 * 1024}' /proc/meminfo)"
  host_os="$(uname -sr)"
fi
git_dirty=false
[[ -n "$(git status --porcelain)" ]] && git_dirty=true

docker compose build gateway
mkdir -p results/embed_latency
docker compose run --rm --no-deps \
  -v "$PWD/scripts:/app/scripts:ro" \
  -v "$PWD/results:/app/results" \
  -e BENCH_HOST_CPU="$host_cpu" \
  -e BENCH_HOST_MEMORY_BYTES="$host_mem" \
  -e BENCH_HOST_OS="$host_os" \
  -e BENCH_DOCKER_SERVER_VERSION="$(docker info --format '{{.ServerVersion}}')" \
  -e BENCH_DOCKER_VM_CPUS="$(docker info --format '{{.NCPU}}')" \
  -e BENCH_DOCKER_VM_MEMORY_BYTES="$(docker info --format '{{.MemTotal}}')" \
  -e BENCH_GIT_COMMIT="$(git rev-parse HEAD)" \
  -e BENCH_GIT_DIRTY="$git_dirty" \
  gateway python scripts/measure_embed_latency.py "$@"
