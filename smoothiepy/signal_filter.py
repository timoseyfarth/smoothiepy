"""
Contains the filters used for signal processing.
"""
from abc import ABC, abstractmethod
from collections import deque
import numpy as np
from typing_extensions import deprecated


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


@deprecated("Filter has no use, why would you use it?")
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
    No weighting is applied, and the average is computed
    as a simple arithmetic mean of the values in the buffer.

    If not enough data points are available to fill the window,
    it computes the average of the available data points.

    :param window_size: The size of the window for averaging.
    :type window_size: int
    """
    def __init__(self, window_size: int):
        super().__init__(window_size)
        self.sum = 0
        self.count = 0

    def _process_next(self, buffer_data: np.array) -> float | int:
        self.sum += buffer_data[-1]

        if self.count < self.window_size:
            self.count += 1
            return self.sum / self.count

        self.sum -= self.latest_removed_buffer_value
        return self.sum / self.window_size


class GaussianAverageFilter1D(Filter1D):
    """
    Implements a Gaussian Average Filter for one-dimensional data.

    Incorporating a Gaussian weighting function applied over a sliding window of data.
    It is used for smoothing data by placing higher importance on values closer more
    recent values of the window while progressively down-weighting values farther away.
    The Gaussian distribution is controlled via the window size and standard deviation parameters.

     If not enough data points are available to fill the window,
    it computes the gaussian average of the available data points with
    a trimmed gaussian filter.

    The filter is only relying on previous data values in the buffer, not future values
     which would result in a delay / offset in the output.

    :param window_size: Size of the sliding window used for the filter.
    :type window_size: int
    :param std_dev: The standard deviation of the Gaussian distribution that determines the spread.
    :type std_dev: float
    :raises ValueError: If std_dev is negative.
    """
    def __init__(self, window_size: int, std_dev: float):
        super().__init__(window_size)
        if std_dev < 0:
            raise ValueError("std_dev must be a non-negative value")

        self.std_dev = std_dev
        self.__gaussian_weights = self.__construct_gaussian_weights()
        self.__gaussian_weights_sum = self.__gaussian_weights.sum()

    def _process_next(self, buffer_data: np.array) -> float | int:
        if len(buffer_data) < self.window_size:
            offset = self.window_size - len(buffer_data)
            weighted_sum = np.sum(buffer_data * self.__gaussian_weights[offset:])
            cur_gaussian_weights_sum = self.__gaussian_weights[offset:].sum()
            return weighted_sum / cur_gaussian_weights_sum

        weighted_sum = np.sum(buffer_data * self.__gaussian_weights)
        return weighted_sum / self.__gaussian_weights_sum

    def __construct_gaussian_weights(self) -> np.array:
        """
        Constructs Gaussian weights based on the specified window size
        and standard deviation.The weights are not centered around zero,
        but rather they are computed from the window size down to zero.
        The most recent data points are given more weight than older ones.

        :return: Computed array of Gaussian weights with length equal to the window size
        :rtype: np.array
        """
        lin_space = np.linspace(self.window_size, 0, self.window_size) \
            if self.window_size % 2 == 0 else np.linspace(self.window_size, 0, self.window_size)
        gaussian = np.exp(-0.5 * (lin_space / self.std_dev) ** 2)
        return gaussian


class MedianAverageFilter1D(Filter1D):
    def _process_next(self, buffer_data: np.array) -> float | int:
        return np.median(buffer_data)
