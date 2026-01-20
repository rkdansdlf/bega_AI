import asyncio
import json
import logging
import os
import random
import signal
import subprocess
import sys
import time
from datetime import datetime
from statistics import mean, median

import httpx
import psutil

# Configuration
API_URL = "http://localhost:8001"
SEARCH_ENDPOINT = f"{API_URL}/search/"
HEALTH_ENDPOINT = f"{API_URL}/health"
DATA_FILE = "AI/data/stress_test_queries.json"
OUTPUT_FILE = "AI/load_test_results.csv"

# Test Parameters
CACHE_SIZES = [0, 256, 1024]  # Reduced set for faster example, extend as needed
WARMUP_DURATION = 60 # Seconds
MEASURE_DURATION = 120 # Seconds
COOLDOWN_DURATION = 10 # Seconds
TARGET_RPS = 20
CONCURRENCY = 10

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("LoadTest")

def load_queries():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["hot"], data["long_tail"]

def zipf_schedule(hot_queries, tail_queries, total_requests):
    """
    Generate a schedule of queries following a rough Zipfian distributon approximation:
    70% from Hot queries (uniform within hot), 30% from Tail queries (uniform within tail).
    """
    schedule = []
    for _ in range(total_requests):
        if random.random() < 0.7:
            schedule.append(random.choice(hot_queries))
        else:
            schedule.append(random.choice(tail_queries))
    return schedule

async def wait_for_server():
    """Wait for the server to be up and running."""
    async with httpx.AsyncClient(timeout=2.0) as client:
        start_time = time.time()
        while time.time() - start_time < 30:
            try:
                resp = await client.get(HEALTH_ENDPOINT)
                if resp.status_code == 200:
                    return True
            except (httpx.ConnectError, httpx.TimeoutException):
                pass
            await asyncio.sleep(1)
    return False

async def run_phase(phase_name, duration, max_rps, queries):
    """Run a single test phase (Warmup / Measure)."""
    logger.info(f"Starting phase: {phase_name} ({duration}s)")
    
    start_time = time.time()
    end_time = start_time + duration
    
    latencies = []
    errors = 0
    request_count = 0
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        while time.time() < end_time:
            batch_start = time.time()
            
            # Fire a batch of requests to approximate Target RPS
            # Simple implementation: fire N requests then sleep remainder of the second
            # Precision is rough but sufficient for this test
            
            batch_queries = [random.choice(queries) for _ in range(max_rps)]
            tasks = []
            
            for q in batch_queries:
                # User requested POST but router defines GET /search/
                # Switching to GET based on code inspection
                tasks.append(client.get(SEARCH_ENDPOINT, params={"q": q, "limit": 10}))

            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            for res in responses:
                request_count += 1
                if isinstance(res, Exception):
                    errors += 1
                elif res.status_code != 200:
                    errors += 1
                else:
                    # Approximation: TTFB or Total Time? httpx response has .elapsed
                    latencies.append(res.elapsed.total_seconds() * 1000) # ms
            
            elapsed = time.time() - batch_start
            sleep_time = 1.0 - elapsed
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
                
    return latencies, errors, request_count

def get_system_metrics(pid):
    try:
        proc = psutil.Process(pid)
        with proc.oneshot():
            cpu_percent = proc.cpu_percent()
            rss_mb = proc.memory_info().rss / (1024 * 1024)
        return cpu_percent, rss_mb
    except psutil.NoSuchProcess:
        return 0, 0

def run_test_cycle(cache_size):
    logger.info(f"--- Starting Cycle: Cache Size {cache_size} ---")
    
    # 1. Start Server
    env = os.environ.copy()
    env["EMBED_QUERY_CACHE_MAX"] = str(cache_size)
    env["EMBED_PROVIDER"] = "local" # Force local for stress testing
    env["LLM_PROVIDER"] = "gemini" # Keep LLM provided if needed, but search endpoint might just retrieve?
    # Note: User wanted "embedding + retrieval only" to isolate LLM latency.
    # The /search endpoint in most RAG apps IS just retrieval. The chat endpoint is RAG.
    # So /search is perfect.
    
    # We need to run uvicorn. assuming 'AI' is CWD.
    # Using 'exec' style to get PID easily
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8001"],
        cwd="AI",
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    try:
        # Wait for server
        loop = asyncio.new_event_loop()
        server_up = loop.run_until_complete(wait_for_server())
        
        if not server_up:
            logger.error("Server failed to start.")
            stdout, stderr = proc.communicate(timeout=1)
            logger.error(f"Server STDOUT: {stdout}")
            logger.error(f"Server STDERR: {stderr}")
            return None
        
        # Prepare Data
        hot, tail = load_queries()
        # Mix them for the phase (Zipf-like probability in run_phase is better)
        # Actually run_phase takes a list and picks randomly. 
        # Let's pre-generate a large pooled list with 70/30 distribution to sample from efficiently
        pool_size = 10000
        query_pool = []
        for _ in range(int(pool_size * 0.7)):
            query_pool.append(random.choice(hot))
        for _ in range(int(pool_size * 0.3)):
            query_pool.append(random.choice(tail))
        random.shuffle(query_pool)

        # 2. Warmup
        loop.run_until_complete(run_phase("Warmup", WARMUP_DURATION, TARGET_RPS, query_pool))
        
        # 3. Measurement
        # Clear metrics
        latencies, errors, count = loop.run_until_complete(run_phase("Measurement", MEASURE_DURATION, TARGET_RPS, query_pool))
        
        # 4. System Metrics
        cpu, rss = get_system_metrics(proc.pid)
        
        # Stats
        p50 = median(latencies) if latencies else 0
        p95 = sorted(latencies)[int(len(latencies)*0.95)] if latencies else 0
        p99 = sorted(latencies)[int(len(latencies)*0.99)] if latencies else 0
        avg_rps = count / MEASURE_DURATION
        
        logger.info(f"Results: Rqs={count}, Err={errors}, P50={p50:.2f}ms, P99={p99:.2f}ms, RSS={rss:.2f}MB")
        
        return {
            "cache_size": cache_size,
            "requests": count,
            "errors": errors,
            "p50_ms": p50,
            "p95_ms": p95,
            "p99_ms": p99,
            "rps": avg_rps,
            "cpu_percent": cpu,
            "rss_mb": rss
        }

    finally:
        # 5. Terminate
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        
        # Cool down
        time.sleep(COOLDOWN_DURATION)

def main():
    if not os.path.exists(DATA_FILE):
        logger.error(f"Data file not found: {DATA_FILE}")
        return

    results = []
    
    # Header
    with open(OUTPUT_FILE, "w") as f:
        f.write("cache_size,requests,errors,rps,p50_ms,p95_ms,p99_ms,cpu_percent,rss_mb\n")

    for size in CACHE_SIZES:
        res = run_test_cycle(size)
        if res:
            results.append(res)
            # Append to file
            with open(OUTPUT_FILE, "a") as f:
                f.write(f"{res['cache_size']},{res['requests']},{res['errors']},{res['rps']:.2f},"
                        f"{res['p50_ms']:.2f},{res['p95_ms']:.2f},{res['p99_ms']:.2f},"
                        f"{res['cpu_percent']:.2f},{res['rss_mb']:.2f}\n")

    logger.info("Load test completed.")

if __name__ == "__main__":
    main()
