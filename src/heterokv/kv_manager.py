"""Controller-owned conservative capacity budgets, not engine cache import/export."""

import asyncio


class Reservations:
    def __init__(self, capacities: dict[str, int]):
        if any(v < 0 for v in capacities.values()):
            raise ValueError("negative capacity")
        self.capacities = dict(capacities)
        self.owners: dict[str, dict[str, int]] = {}
        self.lock = asyncio.Lock()

    def available(self, worker: str) -> int:
        return self.capacities[worker] - sum(r.get(worker, 0) for r in self.owners.values())

    def reserve_locked(self, request_id: str, blocks: dict[str, int]) -> bool:
        if request_id in self.owners:
            raise ValueError("duplicate logical request")
        if not blocks or any(v <= 0 for v in blocks.values()):
            raise ValueError("positive reservation required")
        if any(k not in self.capacities or self.available(k) < v for k, v in blocks.items()):
            return False
        self.owners[request_id] = dict(blocks)
        return True

    async def reserve(self, request_id: str, blocks: dict[str, int]) -> bool:
        async with self.lock:
            return self.reserve_locked(request_id, blocks)

    async def release(self, request_id: str) -> bool:
        async with self.lock:
            return self.owners.pop(request_id, None) is not None

