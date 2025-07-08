"""
Contains the Signal Smoother class which is the main class to smooth signals.
"""
from abc import ABC, abstractmethod
from smoothiepy.filter.basefilter import Filter, Filter1D, Filter2D


class Smoother(ABC):
    """
    Represents an abstract base class for smoothing signals by attaching and managing filters.

    This class is designed as a foundation for implementing specific signal smoothing
    strategies. It provides a structure for managing filters and implementing custom
    signal smoothing logic.

    :ivar filter_list: A list that holds the filters attached to this smoother.
    :type filter_list: list[Filter]
    """
    def __init__(self):
        self.filter_list: list[Filter] = []
        self.is_smoother_built = False

    @abstractmethod
    def attach_filter(self, filter_obj: Filter) -> None:
        """
        Attaches a filter object to the implementing class.

        :param filter_obj: The filter object to attach.
        :type filter_obj: Filter
        """

    def build(self) -> None:
        """
        Builds the final configuration of the smoother.
        """
        self.is_smoother_built = True
        self._build_internal()

    @abstractmethod
    def _build_internal(self) -> None:
        """
        Internal method to build the smoother's configuration.
        """



class Smoother1D(Smoother, ABC):
    """
    Provides one-dimensional signal smoothing functionality.

    This class is an extension of the ``Smoother`` class designed specifically
    for one-dimensional signal data. It maintains a list of 1D filters and allows
    attachment of additional filters which must be of type Filter1D.

    :ivar filter_list: Stores instances of Filter1D for smoothing operations.
    :type filter_list: list[Filter1D]
    """
    def __init__(self):
        super().__init__()
        self.filter_list: list[Filter1D] = []

    def attach_filter(self, filter_obj: Filter) -> None:
        if not isinstance(filter_obj, Filter1D):
            raise TypeError("filter_obj must be an 1D filter")

        self.filter_list.append(filter_obj)


class Smoother1DContinuous(Smoother1D):
    """
    A class for continuously smoothing one-dimensional data.

    Provides functionality to smooth incoming data points in a continuous manner
    using a list of filters.
    The smoothed value is updated as new data points are added.

    :ivar last_filtered_value: Stores the most recent smoothed value.
    :type last_filtered_value: float | int
    """
    def __init__(self):
        super().__init__()
        self.last_filtered_value: float | int = 0.0

    def _build_internal(self) -> None:
        pass

    def add_and_get(self, data: float | int) -> float | int:
        """
        Adds the given data to the signal smoother and returns the filtered value.

        :param data: The numeric value to be added.
        :type data: float | int
        :return: Filtered value after the input has been added.
        :rtype: float | int
        """
        self.add(data)
        return self.get()

    def add(self, data: float | int) -> None:
        """
        Adds a new data point to the signal smoother.

        :param data: The data point to be added.
        :type data: float | int
        """
        temp_filtered_value = data
        for cur_filter in self.filter_list:
            temp_filtered_value = cur_filter.next(temp_filtered_value)

        self.last_filtered_value = temp_filtered_value

    def get(self) -> float | int:
        """
        Retrieves the smoothed value from the signal smoother.

        :return: The smoothed value.
        :rtype: float | int
        """
        return self.last_filtered_value


class Smoother2D(Smoother, ABC):
    """
    Provides two-dimensional signal smoothing functionality.

    This class is an extension of the ``Smoother`` class designed specifically
    for two-dimensional signal data. It maintains a list of 2D filters and allows
    attachment of additional filters which must be of type Filter2D.

    :ivar filter_list: Stores instances of Filter2D for smoothing operations.
    :type filter_list: list[Filter2D]
    """
    def __init__(self):
        super().__init__()
        self.filter_list: list[Filter2D] = []

    def attach_filter(self, filter_obj: Filter) -> None:
        if not isinstance(filter_obj, Filter2D):
            raise TypeError("filter_obj must be an 2D filter")

        self.filter_list.append(filter_obj)


class Smoother2DContinuous(Smoother2D):
    """
        A class for continuously smoothing two-dimensional data.

        Provides functionality to smooth incoming data points in a continuous manner
        using a list of filters.
        The smoothed value is updated as new data points are added.

        :ivar last_filtered_value: Stores the most recent smoothed value.
        :type last_filtered_value: float | int
        """
    def __init__(self):
        super().__init__()
        self.last_filtered_value: tuple[float | int, float | int] = (0.0, 0.0)

    def _build_internal(self) -> None:
        pass

    def add_and_get(self, data_x: float | int, data_y: float | int) \
            -> tuple[float | int, float | int]:
        """
        Adds the given data to the signal smoother and returns the filtered value.

        :param data_x: The first numeric value to be added.
        :type data_x: float | int
        :param data_y: The second numeric value to be added.
        :type data_y: float | int
        :return: Filtered value after the input has been added.
        :rtype: float | int
        """
        self.add(data_x=data_x, data_y=data_y)
        return self.get()

    def add(self, data_x: float | int, data_y: float | int) -> None:
        """
        Adds a new data point to the signal smoother.

         :param data_x: The first numeric value to be added.
        :type data_x: float | int
        :param data_y: The second numeric value to be added.
        :type data_y: float | int
        """
        temp_filtered_value = data_x, data_y
        for cur_filter in self.filter_list:
            temp_filtered_value = cur_filter.next(temp_filtered_value)

        self.last_filtered_value = temp_filtered_value

    def get(self) -> tuple[float | int, float | int]:
        """
        Retrieves the smoothed value from the signal smoother.

        :return: The smoothed value.
        :rtype: float | int
        """
        return self.last_filtered_value


class Smoother1DList(Smoother1D):
    """
    The Smoother1DList class is a specialized implementation for applying 1D smoothing
    on a list-based signal.

    It is particularly suitable for scenarios requiring processing of a signal that is not
    continuous, but you have a batch of data points for smooth through filters.
    """
    def __init__(self):
        super().__init__()
        self.__continuous_smoother = None

    def attach_filter(self, filter_obj: Filter) -> None:
        """
        Attaches a filter object to the instance.

        If the smoother is not built yet, it will not attach the filter to the internal
        continuous smoother. This is because the continuous smoother will be
        built only after the `build` method is called and its state is reset explicitly.

        :param filter_obj: The filter object to be attached.
        :type filter_obj: Filter
        """
        super().attach_filter(filter_obj)
        if not self.is_smoother_built:
            return
        self.__continuous_smoother.attach_filter(filter_obj)

    def _build_internal(self) -> None:
        self.__reset_state()

    def __add_filters_to_continuous_smoother(self) -> None:
        """
        Adds filters from the filter list to the continuous smoother.

        This method iterates over the items in the `filter_list` attribute
        and attaches each filter object to the `__continuous_smoother` object.
        """
        for filter_obj in self.filter_list:
            self.__continuous_smoother.attach_filter(filter_obj)

    def __reset_state(self) -> None:
        """
        Resets the internal state of the instance by reinitializing attributes to their
        default values. This involves deleting the previous smoother and replacing
        it with a new instance to ensure all processing starts with a clean state.
        All filters attached to the smoother are reattached to the new instance.
        """
        del self.__continuous_smoother
        self.__continuous_smoother = Smoother1DContinuous()
        self.__add_filters_to_continuous_smoother()

    def apply_filter(self, signal: list[float | int]) -> list[float | int]:
        """
        Applies a filtering process to a given signal and returns the filtered signal
        after processing through all the attached filters.

        :param signal: A list of numerical data (float or int) representing the signal
                       to be filtered.
        :type signal: list[float | int]
        :return: A list of numerical data (float or int) representing the filtered signal.
        :rtype: list[float | int]
        """
        result = []
        for data in signal:
            filtered = self.__continuous_smoother.add_and_get(data)
            result.append(filtered)
        return result


class Smoother2DList(Smoother2D):
    """
    The Smoother2DList class is a specialized implementation for applying 2D smoothing
    on a list-based signal.

    It is particularly suitable for scenarios requiring processing of a signal that is not
    continuous, but you have a batch of data points for smooth through filters.
    """
    def __init__(self):
        super().__init__()
        self.__continuous_smoother = None

    def attach_filter(self, filter_obj: Filter) -> None:
        """
        Attaches a filter object to the instance.

        If the smoother is not built yet, it will not attach the filter to the internal
        continuous smoother. This is because the continuous smoother will be
        built only after the `build` method is called and its state is reset explicitly.

        :param filter_obj: The filter object to be attached.
        :type filter_obj: Filter
        """
        super().attach_filter(filter_obj)
        if not self.is_smoother_built:
            return
        self.__continuous_smoother.attach_filter(filter_obj)

    def _build_internal(self) -> None:
        self.__reset_state()

    def __add_filters_to_continuous_smoother(self) -> None:
        """
        Adds filters from the filter list to the continuous smoother.

        This method iterates over the items in the `filter_list` attribute
        and attaches each filter object to the `__continuous_smoother` object.
        """
        for filter_obj in self.filter_list:
            self.__continuous_smoother.attach_filter(filter_obj)

    def __reset_state(self) -> None:
        """
        Resets the internal state of the instance by reinitializing attributes to their
        default values. This involves deleting the previous smoother and replacing
        it with a new instance to ensure all processing starts with a clean state.
        All filters attached to the smoother are reattached to the new instance.
        """
        del self.__continuous_smoother
        self.__continuous_smoother = Smoother2DContinuous()
        self.__add_filters_to_continuous_smoother()

    def apply_filter(self, signal_x: list[float | int], signal_y: list[float | int]) \
            -> tuple[list[float | int], list[float | int]]:
        """
        Applies a filtering process to a given signal (x and y direction) and returns
        the filtered signal after processing through all the attached filters.



        :param signal_x: A list of numerical data (float or int) representing the signal
                       to be filtered. This is the x-component of the signal.
        :type signal_x: list[float | int]
        :param signal_y: A list of numerical data (float or int) representing the signal
                       to be filtered. This is the y-component of the signal.
        :type signal_y: list[float | int]
        :return: A list of numerical data (float or int) representing the filtered signal.
        :rtype: list[float | int]
        """
        result_x = []
        result_y = []
        for data_x, data_y in zip(signal_x, signal_y):
            filtered_x, filtered_y = self.__continuous_smoother.add_and_get(data_x, data_y)
            result_x.append(filtered_x)
            result_y.append(filtered_y)
        return result_x, result_y
