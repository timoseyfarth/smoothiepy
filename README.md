![Logo SmoothiePy](https://github.com/user-attachments/assets/74e96edd-efe9-4a76-9f13-499c7f5ea551)

[![Status](https://img.shields.io/badge/status-alpha-lightblue)]()
[![PyPI version](https://img.shields.io/pypi/v/smoothiepy)](https://pypi.org/project/smoothiepy/)
[![Python versions](https://img.shields.io/pypi/pyversions/smoothiepy)](https://pypi.org/project/smoothiepy/)
[![Downloads](https://img.shields.io/pypi/dm/smoothiepy)](https://pypi.org/project/smoothiepy/)
[![License](https://img.shields.io/github/license/timoseyfarth/smoothiepy)](https://github.com/timoseyfarth/smoothiepy/blob/main/LICENSE)
[![Last Commit](https://img.shields.io/github/last-commit/timoseyfarth/smoothiepy)](https://github.com/timoseyfarth/smoothiepy/commits/main)
[![Repo Size](https://img.shields.io/github/repo-size/timoseyfarth/smoothiepy)](https://github.com/timoseyfarth/smoothiepy)

Smooth real-time data streams like eye tracking or sensor input with this lightweight package.

## 📋 Overview

SmoothiePy is a Python library designed for smoothing real-time data streams with minimal latency. 
It provides a collection of filters and smoothers that can be applied to one-dimensional and two-dimensional data, 
making it ideal for applications such as:

- Eye tracking and gaze analysis
- Sensor data processing
- Motion tracking
- Financial data analysis
- Time series preprocessing
- Signal processing

The library is built with a focus on flexibility, performance, and ease of use, allowing you to quickly implement sophisticated data smoothing pipelines.

## 🚀 Installation

```bash
pip install smoothiepy
```

SmoothiePy requires Python 3.10 or later.

## 🏁 Quick Start

Here's a simple example of how to use SmoothiePy to smooth a data stream:

```python
from smoothiepy.smoother.builder import SmootherBuilder
from smoothiepy.filter.filter1d import ExponentialMovingAverageFilter1D

# Create a smoother with an exponential moving average filter
smoother = (
  SmootherBuilder()
  .one_dimensional()
  .continuous()
  .attach_filter(ExponentialMovingAverageFilter1D(alpha=0.2))
  .build()
)

# Process data points
smoother.add(20.0)
print(f"Smoothed value: {smoother.get()}")

smoother.add(60.0)
print(f"Smoothed value: {smoother.get()}")

# Alternatively, use add_and_get to add a value and get the result in one step
smoothed_value = smoother.add_and_get(3.0)
print(f"Smoothed value: {smoothed_value}")
```

## ✨ Features

### Available Filters

SmoothiePy provides a variety of filters:

#### One-Dimensional Filters

- **Offset Filter**: Adds a constant offset to the data
- **Simple Moving Average**: Computes the arithmetic mean over a window
- **Weighted Moving Average**: Applies linearly decreasing weights
- **Gaussian Average**: Applies a Gaussian weighting function
- **Median Average**: Computes the median of values in a window
- **Exponential Moving Average**: Applies exponential weighting
- **Cumulative Moving Average**: Computes the cumulative average
- **Fixation Smooth Filter**: Sort of Deadband filter. Specialized for fixation-like data (e.g., eye tracking)
- **Multi-Pass Moving Average**: Applies multiple passes of a specified moving average type

#### Two-Dimensional Filters

Each 1D filter is also available in a 2D version, allowing you to smooth data in two dimensions (e.g., x-y coordinates).

Many more filters are work in progress, including advanced filters like Kalman filters and more complex multidimensional filters.

### Builder Pattern

SmoothiePy uses a builder pattern to create smoothers, making it easy to configure and chain multiple filters:

```python
from smoothiepy.smoother.builder import SmootherBuilder
from smoothiepy.filter.filter1d import SimpleMovingAverageFilter1D, GaussianAverageFilter1D

# Create a smoother with multiple filters
smoother = (
  SmootherBuilder()
  .one_dimensional()
  .continuous()
  .attach_filter(SimpleMovingAverageFilter1D(window_size=5))
  .attach_filter(GaussianAverageFilter1D(window_size=3, std_dev=1.0))
  .build()
)
```

## 📚 Documentation

For detailed documentation, visit our [GitHub Wiki](https://github.com/timoseyfarth/smoothiepy/wiki).
Coming soon...

### API Reference

#### Filters

- `Filter1D`: Base class for one-dimensional filters
  - `SimpleMovingAverageFilter1D`: Simple arithmetic mean
  - `WeightedMovingAverageFilter1D`: Linearly decreasing weights
  - `GaussianAverageFilter1D`: Gaussian weighting function
  - `MedianAverageFilter1D`: Median of values
  - `ExponentialMovingAverageFilter1D`: Exponential weighting
  - `CumulativeMovingAverageFilter1D`: Cumulative average
  - `FixationSmoothFilter1D`: For fixation-like data
  - `MultiPassMovingAverage1D`: Multiple passes of a specified filter

- `Filter2D`: Base class for two-dimensional filters
  - Various 2D filter implementations

#### Builders

- `SmootherBuilder`: Entry point for creating smoothers
  - `Smoother1DBuilder`: For 1D smoothers
    - `Smoother1DContinuousBuilder`: For continuous 1D smoothers

## 📄 License

This project is licensed under the GNU General Public License v3.0 (GPL-3.0) - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- NumPy for efficient numerical operations

## 📬 Contact

Timo Seyfarth - timo@seyfarth.dev
