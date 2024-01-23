
import threading


class LockQueue:

    def __init__(self):
        self._lock = threading.Lock()
        self._queue = []

    def enqueue(self, item):
        with self._lock:
            self._queue.append(item)

    def dequeue(self):
        with self._lock:
            item = self._queue.pop(0)
        return item


    def size(self):
        with self._lock:
            size = len(self._queue)
        return size