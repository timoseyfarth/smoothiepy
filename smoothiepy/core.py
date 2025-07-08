"""
Main module for the application.
"""
from smoothiepy import MedianAverageFilter1D
from smoothiepy.smoother.builder import SmootherBuilder

def main() -> None:
    """
    Main function to demonstrate and test the usage of the Signal Smoother.
    """
    smoother = (
        SmootherBuilder()
        .one_dimensional()
        .list_based()
        .attach_filter(MedianAverageFilter1D(window_size=2))
        .build()
    )

    print("Init finished")

    data_list = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]

    result_list = smoother.apply_filter(data_list)

    print(f"Original data: {data_list}")
    print(f"Filtered data: {result_list}")


if __name__ == "__main__":
    main()
