"""Request-wide budget propagated into synchronous tool calls."""
from contextvars import ContextVar
import time
budget_end = ContextVar("agent_budget_end", default=None)
def remaining(default=240.0):
    end=budget_end.get()
    value=default if end is None else min(default,end-time.monotonic())
    if value<=0: raise TimeoutError("请求总时间预算已用尽")
    return max(0.1,value)
