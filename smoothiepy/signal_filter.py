"""
Contains the filters used for signal processing.
"""
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
    :param window_size: The size of the window for the filter.
    :type window_size: int
    :ivar window_size: The size of the window for the filter.
    :type window_size: int
    """
    def __init__(self, window_size: int):
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
        print(self.buffer)
        print(self.latest_removed_buffer_value)
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


class UselessFilter1D(Filter1D):
    """
    A filter that does not perform any filtering.
    It simply returns the input data as is.
    """
    def __init__(self):
        super().__init__(window_size=1)

    def _process_next(self, buffer_data: np.array) -> float | int:
        return buffer_data[0]


class OffsetFilter1D(Filter1D):
    """
    A filter that applies a constant offset to the input data.

    :param offset: The constant value to be added to the input data.
    :type offset: float | int
    """
    def __init__(self, offset: float | int):
        super().__init__(window_size=1)
        self.offset = offset

    def _process_next(self, buffer_data: np.array) -> float | int:
        return buffer_data[0] + self.offset


class AverageFilter1D(Filter1D):
    """
    A filter that computes the average of the input data over a specified window size.

    :param window_size: The size of the window for averaging.
    :type window_size: int
    """
    def __init__(self, window_size: int):
        super().__init__(window_size)
        self.sum = 0
        self.count = 0

    def _process_next(self, buffer_data: np.array) -> float | int:
        """
        Computes the average of the current buffer data.

        :param buffer_data: The current buffer data.
        :type buffer_data: list[float | int]
        :return: The average of the buffer data.
        :rtype: float | int
        """
        self.sum += buffer_data[-1]

        if self.count < self.window_size:
            self.count += 1
            return self.sum / self.count

        self.sum -= self.latest_removed_buffer_value
        return self.sum / self.window_size
