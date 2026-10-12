"""Bounded worker pool; a timed-out job retains its slot until it exits."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
import os
import time
from rag_agent.deadline import budget_end
class HttpRuntime:
    def __init__(self):
        workers=max(1,min(8,int(os.getenv("AGENT_WORKERS","4"))))
        self.pool=ThreadPoolExecutor(max_workers=workers,thread_name_prefix="agent")
        self.slots=asyncio.Semaphore(workers)
    async def call(self,handler,request):
        timeout=float(os.getenv("AGENT_REQUEST_TIMEOUT","220"))
        await asyncio.wait_for(self.slots.acquire(),timeout=1)
        token=budget_end.set(time.monotonic()+timeout)
        try:
            context=copy_context()
            future=asyncio.get_running_loop().run_in_executor(self.pool,context.run,handler,request)
        except BaseException:
            self.slots.release()
            raise
        finally:
            budget_end.reset(token)
        def completed(done):
            self.slots.release()
            if not done.cancelled():done.exception()
        future.add_done_callback(completed)
        return await asyncio.wait_for(asyncio.shield(future),timeout=timeout)
