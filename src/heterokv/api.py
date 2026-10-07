import asyncio
import json
from contextlib import aclosing, asynccontextmanager
from collections import Counter
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse, Response, StreamingResponse
from pydantic import BaseModel
from starlette.background import BackgroundTask

from heterokv.contracts import Request, Status
from heterokv.controller import Rejected
from heterokv.runtime import build_controller, refresh_workers
from heterokv.artifacts import validate_run


class Generation(BaseModel):
    request: Request
    prompt_token_ids: list[int] | None = None


def create_app(config):
    controller = build_controller(config)
    terminal = Counter()

    @asynccontextmanager
    async def lifespan(app):
        task = asyncio.create_task(refresh_workers(controller, config))
        yield
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    app = FastAPI(title="HeteroKV", lifespan=lifespan)
    app.state.controller = controller

    @app.get("/", include_in_schema=False)
    async def index():
        return RedirectResponse(url="/docs")

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon():
        return Response(status_code=204)

    @app.get("/health")
    async def health():
        return {"backend": config.get("backend", "mock"),
                "evidence_kind": controller.adapter.evidence_kind,
                "inflight": len(controller.reservations.owners),
                "supports_real_pd": False}

    @app.get("/metrics")
    async def metrics():
        lines = ['# TYPE heterokv_terminal_total counter']
        lines.extend(f'heterokv_terminal_total{{status="{status}"}} {count}'
                     for status, count in sorted(terminal.items()))
        return Response("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")

    @app.get("/results/latest")
    async def latest_result():
        """Most recent valid saved benchmark, distinct from this server's live counters."""
        root = Path(config.get("runs_dir", "results/runs"))
        if root.is_dir():
            for run in sorted(root.iterdir(), reverse=True):
                if run.is_dir() and (run / "summary.json").is_file():
                    validation = validate_run(run)
                    if validation["valid"]:
                        summary = json.loads((run / "summary.json").read_text())
                        return {"source": "saved benchmark", "run_id": run.name,
                                "evidence_kind": summary["evidence_kind"], "summary": summary}
        raise HTTPException(404, "No valid saved benchmark runs found")

    @app.post("/generate")
    async def generate(body: Generation):
        try:
            plan = await controller.admit(body.request)
        except Rejected as exc:
            terminal["REJECTED"] += 1
            raise HTTPException(429, str(exc)) from exc

        async def events():
            # Direct pull streaming provides backpressure without an unbounded token queue.
            try:
                async with aclosing(controller.stream(body.request, body.prompt_token_ids,
                                                       admitted=plan)) as stream:
                    async for event in stream:
                        yield "data: " + event.model_dump_json() + "\n\n"
            except TimeoutError:
                yield "data: " + json.dumps({"error": "TIMED_OUT"}) + "\n\n"
            except Exception:
                yield "data: " + json.dumps({"error": "FAILED"}) + "\n\n"
            finally:
                await controller.finish(body.request, Status.CANCELLED)
                terminal[controller.states[body.request.request_id].value] += 1

        return StreamingResponse(events(), media_type="text/event-stream",
                                 background=BackgroundTask(controller.finish, body.request,
                                                           Status.CANCELLED))

    return app
