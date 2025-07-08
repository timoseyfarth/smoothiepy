"""
Contains the helper classes to build a signal smoother.
"""
from smoothiepy.filter.basefilter import Filter1D, Filter2D
from smoothiepy.smoother.smoother import Smoother1DContinuous, Smoother2DContinuous, Smoother1DList, Smoother2DList


class SmootherBuilder:
    """
    Provides utilities for building smoother objects for data.

    This class provides methods to construct smoother objects based on
    specified dimensional requirements. It currently supports
    one-dimensional smoother creation.
    """
    @staticmethod
    def one_dimensional() -> 'Smoother1DBuilder':
        return Smoother1DBuilder()

    @staticmethod
    def two_dimensional() -> 'Smoother2DBuilder':
        return Smoother2DBuilder()

    def set_dimensions(self, dimensions: int) -> 'Smoother1DBuilder | Smoother2DBuilder':
        if dimensions == 1:
            return self.one_dimensional()
        if dimensions == 2:
            return self.two_dimensional()

        raise ValueError("Unsupported dimensions. Currently only 1D and 2D is supported.")


class Smoother1DBuilder:
    """
    Provides a builder for creating 1D signal smoother instances.

    This class offers static methods to initialize builders for smoothing
    operations tailored to different types of data. It simplifies the
    creation of signal smoother objects by providing predefined builder
    methods.
    """
    @staticmethod
    def continuous() -> 'Smoother1DContinuousBuilder':
        return Smoother1DContinuousBuilder()

    @staticmethod
    def list_based() -> 'Smoother1DListBuilder':
        return Smoother1DListBuilder()


class Smoother2DBuilder:
    """
    Provides a builder for creating 2D signal smoother instances.

    This class offers static methods to initialize builders for smoothing
    operations tailored to different types of data. It simplifies the
    creation of signal smoother objects by providing predefined builder
    methods.
    """
    @staticmethod
    def continuous() -> 'Smoother2DContinuousBuilder':
        return Smoother2DContinuousBuilder()

    @staticmethod
    def list_based() -> 'Smoother2DListBuilder':
        return Smoother2DListBuilder()


class Smoother1DContinuousBuilder:
    """
    A builder class for creating 1D continuous signal smoother.

    This class facilitates the creation and configuration of a 1D continuous
    signal smoother by allowing filters to be attached and then constructing
    the smoother object.
    """
    def __init__(self):
        self.__smoother = Smoother1DContinuous()

    def attach_filter(self, filter_obj: Filter1D) -> 'Smoother1DContinuousBuilder':
        self.__smoother.attach_filter(filter_obj)
        return self

    def build(self) -> Smoother1DContinuous:
        self.__smoother.build()
        return self.__smoother


class Smoother2DContinuousBuilder:
    """
    A builder class for creating 2D continuous signal smoother.

    This class facilitates the creation and configuration of a 2D continuous
    signal smoother by allowing filters to be attached and then constructing
    the smoother object.
    """
    def __init__(self):
        self.__smoother = Smoother2DContinuous()

    def attach_filter(self, filter_obj: Filter2D) -> 'Smoother2DContinuousBuilder':
        self.__smoother.attach_filter(filter_obj)
        return self

    def build(self) -> Smoother2DContinuous:
        self.__smoother.build()
        return self.__smoother


class Smoother1DListBuilder:
    """
    A builder class for creating a 1D signal smoother that operates on lists.

    This class allows the attachment of filters and builds a smoother that
    processes data from separate lists, enabling more flexible data handling.
    """
    def __init__(self):
        self.__smoother = Smoother1DList()

    def attach_filter(self, filter_obj: Filter1D) -> 'Smoother1DListBuilder':
        self.__smoother.attach_filter(filter_obj)
        return self

    def build(self) -> Smoother1DList:
        self.__smoother.build()
        return self.__smoother


class Smoother2DListBuilder:
    """
    A builder class for creating a 2D signal smoother that operates on lists.

    This class allows the attachment of filters and builds a smoother that
    processes data from separate lists, enabling more flexible data handling.
    """
    def __init__(self):
        self.__smoother = Smoother2DList()

    def attach_filter(self, filter_obj: Filter2D) -> 'Smoother2DListBuilder':
        self.__smoother.attach_filter(filter_obj)
        return self

    def build(self) -> Smoother2DList:
        self.__smoother.build()
        return self.__smoother
