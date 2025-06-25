"""
Contains the filters used for signal processing.
"""
from abc import ABC, abstractmethod
from collections import deque


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
        self.buffer = None

    @abstractmethod
    def next(self, data: float | int) -> float | int:
        """
        Processes the next data point and returns the filtered value.

        :param data: The next data point to be processed.
        :type data: float | int
        :return: The filtered value.
        :rtype: float | int
        """

    def set_buffer_size(self, amount: int) -> None:
        """
        Sets the maximum size for the internal buffer by initializing a deque with
        a specified maximum length. This method adjusts the capacity of the buffer
        to handle a fixed number of elements.

        :param amount: The maximum number of elements the buffer can contain.
        :type amount: int
        """
        self.buffer = deque(maxlen=amount)


class UselessFilter1D(Filter1D):
    """
    A filter that does not perform any filtering.
    It simply returns the input data as is.
    """
    def __init__(self):
        super().__init__(window_size=1)

    def next(self, data: float | int) -> float | int:
        return data


class OffsetFilter1D(Filter1D):
    """
    A filter that applies a constant offset to the input data.

    :param offset: The constant value to be added to the input data.
    :type offset: float | int
    """
    def __init__(self, offset: float | int):
        super().__init__(window_size=1)
        self.offset = offset

    def next(self, data: float | int) -> float | int:
        return data + self.offset
