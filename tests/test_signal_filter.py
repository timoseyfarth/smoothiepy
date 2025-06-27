import numpy as np
import pytest

from smoothiepy.signal_filter import (OffsetFilter1D, AverageFilter1D, GaussianAverageFilter1D,
                                      MedianAverageFilter1D, ExponentialMovingAverageFilter1D,
                                      FixationSmoothFilter1D)


class TestOffsetFilter1D:
    @pytest.fixture
    def offset_filter_positive(self):
        return OffsetFilter1D(offset=5)

    @pytest.fixture
    def offset_filter_negative(self):
        return OffsetFilter1D(offset=-3)

    @pytest.fixture
    def offset_filter_zero(self):
        return OffsetFilter1D(offset=0)

    @pytest.fixture
    def offset_filter_float(self):
        return OffsetFilter1D(offset=2.5)

    @pytest.mark.parametrize(
        'data, expected',
        [
            (10, 15),
            (20, 25),
            (30, 35),
            (0, 5),
            (-10, -5),
        ]
    )
    def test_positive_offset(self, offset_filter_positive, data, expected):
        assert offset_filter_positive.next(data) == expected

    @pytest.mark.parametrize(
        'data, expected',
        [
            (10, 7),
            (20, 17),
            (30, 27),
            (0, -3),
            (-10, -13),
        ]
    )
    def test_negative_offset(self, offset_filter_negative, data, expected):
        assert offset_filter_negative.next(data) == expected

    @pytest.mark.parametrize(
        'data, expected',
        [
            (10, 10),
            (20, 20),
            (30, 30),
            (0, 0),
            (-10, -10),
        ]
    )
    def test_zero_offset(self, offset_filter_zero, data, expected):
        assert offset_filter_zero.next(data) == expected

    @pytest.mark.parametrize(
        'data, expected',
        [
            (10, 12.5),
            (20, 22.5),
            (30, 32.5),
            (0, 2.5),
            (-10, -7.5),
        ]
    )
    def test_float_offset(self, offset_filter_float, data, expected):
        assert offset_filter_float.next(data) == expected


class TestAverageFilter1D:
    @pytest.fixture
    def average_filter_w2(self):
        return AverageFilter1D(window_size=2)

    @pytest.fixture
    def average_filter_w3(self):
        return AverageFilter1D(window_size=3)

    @pytest.fixture
    def average_filter_w10(self):
        return AverageFilter1D(window_size=10)

    def test_negative_window_size(self):
        with pytest.raises(ValueError):
            AverageFilter1D(window_size=-1)

    @pytest.mark.parametrize(
        'data_list, expected_list',
        [
            ([10], [10.0]),
            ([10, 20], [10.0, 15.0]),
            ([10, 20, 30, 40, 50], [10.0, 15.0, 25.0, 35.0, 45.0]),
            ([10, -10, 20, -20, 30], [10.0, 0.0, 5.0, 0.0, 5.0]),
            ([0, 0, 0], [0.0, 0.0, 0.0]),
            ([0.5, 1.5, 2.5], [0.5, 1.0, 2.0]),
            ([0.5, -1.5, 2.5], [0.5, -0.5, 0.5]),
        ]
    )
    def test_average_filter_w2(self, average_filter_w2, data_list, expected_list):
        for data, expected in zip(data_list, expected_list):
            assert average_filter_w2.next(data) == expected

    @pytest.mark.parametrize(
        'data_list, expected_list',
        [
            ([10], [10.0]),
            ([10, 20], [10.0, 15.0]),
            ([10, 20, 30, 40, 50], [10.0, 15.0, 20.0, 30.0, 40.0]),
            ([10, -10, 30, -20, 20], [10.0, 0.0, 10.0, 0.0, 10.0]),
            ([0, 0, 0], [0.0, 0.0, 0.0]),
            ([0.5, 2.75, 2.75], [0.5, 1.625, 2.0]),
            ([0.5, -1.5, 2.5], [0.5, -0.5, 0.5]),
        ]
    )
    def test_average_filter_w3(self, average_filter_w3, data_list, expected_list):
        for data, expected in zip(data_list, expected_list):
            assert average_filter_w3.next(data) == expected
    
    @pytest.mark.parametrize(
        'data_list, expected_list',
        [
            ([10], [10.0]),
            ([10, 20], [10.0, 15.0]),
            ([10, 20, 30, 40, 50], [10.0, 15.0, 20.0, 25.0, 30.0]),
            ([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120],
             [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 65.0, 75.0]),
        ]
    )
    def test_average_filter_w10(self, average_filter_w10, data_list, expected_list):
        for data, expected in zip(data_list, expected_list):
            assert average_filter_w10.next(data) == expected


class TestGaussianAverageFilter1D:
    @pytest.fixture
    def gauss_filter_std1(self):
        return GaussianAverageFilter1D(window_size=3, std_dev=1.0)

    @pytest.fixture
    def gauss_filter_std0_5(self):
        return GaussianAverageFilter1D(window_size=5, std_dev=0.5)

    @pytest.fixture
    def gauss_filter_std2(self):
        return GaussianAverageFilter1D(window_size=5, std_dev=2.0)

    def test_invalid_window_size(self):
        with pytest.raises(ValueError):
            GaussianAverageFilter1D(window_size=0, std_dev=1.0)

    def test_invalid_std_dev(self):
        with pytest.raises(ValueError):
            GaussianAverageFilter1D(window_size=3, std_dev=-1.0)

    @pytest.mark.parametrize(
        "data_list, expected_approx",
        [
            ([1], [1.0]),
            ([1, 2], [1.0, pytest.approx((0.3246 * 1 + 1 * 2) / (1 + 0.3246), rel=1e-2)]),
            ([1, 2, 3], [1.0, pytest.approx((0.3246 * 1 + 1 * 2) / (1 + 0.3246), rel=1e-2), pytest.approx((0.011109 * 1 + 0.3246 * 2 + 1*3) / (1 + 0.3246 + 0.0111), rel=1e-2)]),
        ]
    )
    def test_gaussian_average_basic(self, gauss_filter_std1, data_list, expected_approx):
        results = []
        for data in data_list:
            results.append(gauss_filter_std1.next(data))
        for result, expected in zip(results, expected_approx):
            assert result == expected

    def test_gaussian_weights_sum_is_correct(self, gauss_filter_std2):
        weights = gauss_filter_std2._GaussianAverageFilter1D__gaussian_weights
        assert isinstance(weights, np.ndarray)
        assert np.isclose(weights.sum(), gauss_filter_std2._GaussianAverageFilter1D__gaussian_weights_sum)

    def test_gaussian_weights_shape_and_monotonicity(self, gauss_filter_std0_5):
        weights = gauss_filter_std0_5._GaussianAverageFilter1D__gaussian_weights
        assert len(weights) == 5
        assert all(weights[i] <= weights[i + 1] for i in range(len(weights) - 1))

    def test_filter_converges_to_mean_on_constant_input(self):
        gauss_filter = GaussianAverageFilter1D(window_size=10, std_dev=1.0)
        data = [5.0] * 20
        for i, val in enumerate(data):
            result = gauss_filter.next(val)
            if i >= 9:
                assert result == pytest.approx(5.0, rel=1e-3)

    def test_construct_gaussian_weights_correctness(self):
        window_size = 4
        std_dev = 1.0
        gauss_filter = GaussianAverageFilter1D(window_size=window_size, std_dev=std_dev)

        expected_lin_space = np.linspace(window_size, 0, window_size)
        expected_weights = np.exp(-0.5 * (expected_lin_space / std_dev) ** 2)

        actual_weights = gauss_filter._GaussianAverageFilter1D__construct_gaussian_weights()

        np.testing.assert_allclose(actual_weights, expected_weights, rtol=1e-8)


class TestMedianFilter1D:
    @pytest.fixture
    def median_filter_w3(self):
        return MedianAverageFilter1D(window_size=3)

    @pytest.fixture
    def median_filter_w5(self):
        return MedianAverageFilter1D(window_size=5)

    def test_invalid_window_size(self):
        with pytest.raises(ValueError):
            MedianAverageFilter1D(window_size=0)

    @pytest.mark.parametrize(
        'data_list, expected_list',
        [
            ([10], [10.0]),
            ([10, 20], [10.0, 15.0]),
            ([10, 20, 30], [10.0, 15.0, 20.0]),
            ([10, 20, 30, 40], [10.0, 15.0, 20.0, 30.0]),
            ([10, 20, 30, 40, 50], [10.0, 15.0, 20.0, 30.0, 40.0]),
            ([10, -10, -20], [10.0, 0.0, -10.0]),
            ([1, 2, 3], [1.0, 1.5, 2.0]),
        ]
    )
    def test_median_filter_w3(self, median_filter_w3, data_list, expected_list):
        for data, expected in zip(data_list, expected_list):
            assert median_filter_w3.next(data) == expected

    @pytest.mark.parametrize(
        'data_list, expected_list',
        [
            ([10], [10.0]),
            ([10, 20, 30], [10.0, 15.0, 20.0]),
            ([10, -10, -20], [10.0, 0.0, -10.0]),
            ([10, 20, 30, 40, 50], [10.0, 15.0, 20.0, 25.0, 30.0]),
            ([1, 2], [1, 1.5]),
        ]
    )
    def test_median_filter_w5(self, median_filter_w5, data_list, expected_list):
        for data in data_list:
            assert median_filter_w5.next(data) == expected_list.pop(0)


class TestExponentialMovingAverageFilter1D:
    def test_invalid_alpha_below_0(self):
        with pytest.raises(ValueError):
            ExponentialMovingAverageFilter1D(alpha=-0.1)

    def test_invalid_alpha_above_1(self):
        with pytest.raises(ValueError):
            ExponentialMovingAverageFilter1D(alpha=1.1)

    @pytest.mark.parametrize(
        'alpha, data_list, expected_list',
        [
            (0.5, [10, 20], [10.0, 15.0]),
            (0.5, [10, 20, 30], [10.0, 15.0, 22.5]),
            (0, [10, 30], [10.0, 10.0]),
            (1, [10, 30], [10.0, 30.0]),
            (0.2, [20, 60], [20.0, 28.0]),
            (0.8, [10, 30], [10.0, 26.0]),
        ]
    )
    def test_ema_filter(self, alpha, data_list, expected_list):
        ema_filter = ExponentialMovingAverageFilter1D(alpha=alpha)
        for data, expected in zip(data_list, expected_list):
            assert ema_filter.next(data) == expected


class TestFixationSmoothingFilter1D:
    @pytest.fixture
    def fixation_filter(self):
        return FixationSmoothFilter1D(window_size=3, threshold=4)

    def test_invalid_window_size(self):
        with pytest.raises(ValueError):
            FixationSmoothFilter1D(window_size=0, threshold=4)

    @pytest.mark.parametrize(
        "data_list, expected_list",
        [
            ([500, 10, 100, 7, 300], [500, 10, 100, 7, 300]),
            ([500, 10, 200, 30, 400], [500.0, 10.0, 200.0, 30.0, 400.0]),
            ([500, -10, -200, -30, -400], [500.0, -10.0, -200.0, -30.0, -400.0]),
            ([500, 10, 200], [500.0, 10.0, 200.0]),
            ([500], [500.0]),
            ([500.0, -10.0], [500.0, -10.0]),
        ]
    )
    def test_fixation_smoothing_jumpy_data(self, fixation_filter, data_list, expected_list):
        results = []
        for data in data_list:
            results.append(fixation_filter.next(data))
        assert results == expected_list

    @pytest.mark.parametrize(
        "data_list, expected_list",
        [
            ([500, 501, 503, 500, 499], [500.0, 500.0, 500.0, 500.0, 500.0]),
            ([500, 501, 502, 503, 502], [500.0, 500.0, 500.0, 500.0, 500.0]),
            ([500, 501, 499, 500, 501, 499, 400, 402], [500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 400, 402.0]),
            ([500, 501, 499, 500, 501, 499, 400, 402, 399, 400, 401],
                [500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 400, 402.0, 399, pytest.approx(400.1111, rel=1e-2), pytest.approx(400.1111, rel=1e-2)]),

        ]
    )
    def test_fixation_smoothing_fixation_data(self, fixation_filter, data_list, expected_list):
        results = []
        for data in data_list:
            results.append(fixation_filter.next(data))
        assert results == expected_list
