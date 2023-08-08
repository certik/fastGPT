"""See https://realpython.com/python-timer/"""


from dataclasses import dataclass, field
import time
from typing import Callable, Optional


class TimerError(Exception):
    """custom Exception for Timer errors"""


@dataclass
class Timer:
    """nestable"""
    text        : str = "Elapsed time: {:0.4f} seconds"
    logger      : Callable[[str], None] = print
    _start_time : Optional[float] = \
        field(default=None, init=False, repr=False)

    def start(self) -> None:
        self._start_time = time.perf_counter()

    def stop(self) -> float:
        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None
        self.logger(self.text.format(elapsed_time))
        return elapsed_time

    # Protocol methods for context manager:
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc_info):
        self.stop()
