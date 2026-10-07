"""Poll engine liveness on the controller host. Does not measure cache capacity or GPU load.

Use dedicated engines and premeasured conservative cache budgets. A live HTTP endpoint
does not establish connector or model compatibility. This only refreshes liveness timestamps.
"""
import argparse
import asyncio
import json
import os
import platform
import tempfile
import time
from pathlib import Path

import httpx
import yaml


async def run(config, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(timeout=1) as client:
        while True:
            async def check(worker):
                try:
                    response = await client.get(worker["endpoint"].rstrip("/") + "/health")
                    healthy = response.status_code == 200
                except httpx.HTTPError:
                    healthy = False
                return {"worker_id": worker["worker_id"], "monotonic_time": time.monotonic(),
                        "healthy": healthy}
            samples = await asyncio.gather(*(check(w) for w in config["workers"]))
            fd, temporary = tempfile.mkstemp(dir=destination.parent, prefix="health-")
            with os.fdopen(fd, "w") as handle:
                json.dump({"hostname": platform.node(), "workers": samples}, handle)
            os.replace(temporary, destination)
            await asyncio.sleep(.2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    asyncio.run(run(config, Path(config["telemetry_file"])))
