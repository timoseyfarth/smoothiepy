"""
Main module for the application.
"""
from smoothiepy.signal_filter import OffsetFilter1D
from smoothiepy.signal_smoother_builder import SmootherBuilder


def main() -> None:
    """
    Main function to demonstrate and test the usage of the Signal Smoother.
    """
    smoother = (
        SmootherBuilder()
        .one_dimensional()
        .set_continuous()
        .attach_filter(OffsetFilter1D(offset=2.0))
        .attach_filter(OffsetFilter1D(offset=5))
        .build()
    )

    print("Init finished")

    smoother.add(42.0)
    print(f"Smoothed value 1: {smoother.get()}")
    smoother.add(68.0)
    print(f"Smoothed value 2: {smoother.get()}")
    smoother.add(100.0)
    print(f"Smoothed value 3: {smoother.get()}")
    smoother.add(3.14)
    print(f"Smoothed value 4: {smoother.get()}")


if __name__ == "__main__":
    main()
