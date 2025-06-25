"""
Contains the helper classes to build a signal smoother.
"""
from src.signal_filter import Filter1D
from src.signal_smoother import Smoother1DContinuous


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

    def set_dimensions(self, dimensions: int) -> 'Smoother1DBuilder':
        if dimensions == 1:
            return self.one_dimensional()

        raise ValueError("Unsupported dimensions. Currently only 1D is supported.")


class Smoother1DBuilder:
    """
    Provides a builder for creating 1D signal smoother instances.

    This class offers static methods to initialize builders for smoothing
    operations tailored to different types of data. It simplifies the
    creation of signal smoother objects by providing predefined builder
    methods.
    """
    @staticmethod
    def set_continuous() -> 'Smoother1DContinuousBuilder':
        return Smoother1DContinuousBuilder()


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


# class SignalSmootherBuilder1DList:
#     def __init__(self):
#         self.smoother = SignalSmootherSeparateLists1D()
#
#     def attach_filter(self, filter_obj: BaseFilter1D):
#         self.smoother.attach_filter(filter_obj)
#
#     def build(self):
#         self.smoother.build()
#         return self.smoother
