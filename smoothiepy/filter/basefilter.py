"""
Base module that defines the abstract base class for all signal filters.
"""
import enum
from abc import ABC, abstractmethod
from collections import deque

import numpy as np


class Filter(ABC):
    """
    Abstract base class for all signal filters.
    """


class Filter1D(Filter, ABC):
    """
    Abstract base class for all one-dimensional signal filters.
    :param window_size: The size of the window for the filter. Must be a positive integer.
    :type window_size: int
    :ivar window_size: The size of the window for the filter.
    :type window_size: int
    :raises ValueError: If window_size is not a positive integer.
    """
    def __init__(self, window_size: int):
        if window_size <= 0:
            raise ValueError("window_size must be greater than 0")
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)
        self.latest_removed_buffer_value: float | int = 0.0

    def next(self, data: float | int) -> float | int:
        """
        Processes the next data point by adding it to the buffer
        and calling the internal processing method.

        :param data: The next data point to be processed.
        :type data: float | int
        :return: The filtered value.
        :rtype: float | int
        """
        self.latest_removed_buffer_value = self.buffer[0] if self.buffer else 0.0
        self.buffer.append(data)
        return self._process_next(np.array(self.buffer))

    @abstractmethod
    def _process_next(self, buffer_data: np.array) -> float | int:
        """
        Processes the next data point using the current buffer data.

        :param buffer_data: The current buffer data.
        :type buffer_data: list[float | int]
        :return: The processed value.
        :rtype: float | int
        """


class MovingAverageType(enum.Enum):
    """
    Enum for different types of moving averages.

    This enum defines the types of moving averages that can be applied to signals.
    """
    SIMPLE = "simple"
    WEIGHTED = "weighted"
    GAUSSIAN = "gaussian"
    MEDIAN = "median"
    EXPONENTIAL = "exponential"
    CUMULATIVE = "cumulative"
