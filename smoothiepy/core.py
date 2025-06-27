"""
Main module for the application.
"""
from signal_smoother_builder import SmootherBuilder
from signal_filter import MedianAverageFilter1D

def main() -> None:
    """
    Main function to demonstrate and test the usage of the Signal Smoother.
    """
    smoother = (
        SmootherBuilder()
        .one_dimensional()
        .set_continuous()
        .attach_filter(MedianAverageFilter1D(window_size=3))
        .build()
    )

    print("Init finished")

    smoother.add(40.0)
    print(f"Smoothed value 1: {smoother.get()}")
    smoother.add(60.0)
    print(f"Smoothed value 2: {smoother.get()}")
    smoother.add(100.0)
    print(f"Smoothed value 3: {smoother.get()}")
    smoother.add(3)
    print(f"Smoothed value 4: {smoother.get()}")


if __name__ == "__main__":
    main()
