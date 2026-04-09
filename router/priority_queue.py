"""
Priority Request Queue

Thread-safe priority queue for inference requests.  Lower ``priority``
values are dequeued first (0 = highest priority).  Expired requests
(past their ``deadline``) are dropped automatically on dequeue.

Usage::

    queue = PriorityRequestQueue()
    queue.enqueue(InferenceRequest(id="r1", payload={"prompt": "hi"}, priority=1))
    request = queue.dequeue()
"""
from __future__ import annotations

import heapq
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(order=False)
class InferenceRequest:
    """A single inference request with priority and deadline metadata.

    Parameters
    ----------
    id : str
        Unique request identifier.
    payload : dict
        Arbitrary request payload (prompt, images, parameters, etc.).
    priority : int
        Lower values are served first. Default 5 (medium).
    deadline : float
        Unix timestamp after which the request should be discarded.
        Default 0.0 means no deadline.
    enqueue_time : float
        Unix timestamp when the request entered the queue (auto-filled).
    """

    id: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    priority: int = 5
    deadline: float = 0.0
    enqueue_time: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not self.id:
            self.id = uuid.uuid4().hex

    # heapq needs a total ordering — sort by (priority, enqueue_time).
    def __lt__(self, other: "InferenceRequest") -> bool:
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.enqueue_time < other.enqueue_time

    def is_expired(self) -> bool:
        """Return ``True`` if the request has passed its deadline."""
        if self.deadline <= 0.0:
            return False
        return time.time() > self.deadline


class PriorityRequestQueue:
    """Thread-safe min-heap priority queue for inference requests.

    All public methods acquire a ``threading.Lock`` so the queue can be
    shared safely across worker threads.
    """

    def __init__(self) -> None:
        self._heap: List[InferenceRequest] = []
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enqueue(self, request: InferenceRequest) -> None:
        """Add a request to the queue."""
        with self._lock:
            heapq.heappush(self._heap, request)

    def dequeue(self) -> Optional[InferenceRequest]:
        """Remove and return the highest-priority non-expired request.

        Expired entries are silently discarded.  Returns ``None`` when the
        queue is empty (or all remaining entries are expired).
        """
        with self._lock:
            while self._heap:
                request = heapq.heappop(self._heap)
                if not request.is_expired():
                    return request
            return None

    def peek(self) -> Optional[InferenceRequest]:
        """Return the highest-priority request without removing it.

        Skips expired entries (which are removed as a side-effect).
        """
        with self._lock:
            while self._heap:
                if self._heap[0].is_expired():
                    heapq.heappop(self._heap)
                else:
                    return self._heap[0]
            return None

    def drop_expired(self) -> int:
        """Remove all expired requests and return how many were dropped."""
        with self._lock:
            before = len(self._heap)
            self._heap = [r for r in self._heap if not r.is_expired()]
            heapq.heapify(self._heap)
            return before - len(self._heap)

    def __len__(self) -> int:
        with self._lock:
            return len(self._heap)

    def __bool__(self) -> bool:
        with self._lock:
            return len(self._heap) > 0
