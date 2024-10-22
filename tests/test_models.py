"""Tests for statistics functions within the Model layer."""

import numpy as np
import numpy.testing as npt
import pytest
import math
from unittest.mock import Mock
from pathlib import Path
import pytest

@pytest.mark.parametrize(
    "test, expected",
    [
        ([ [0, 0], [0, 0], [0, 0] ], [0, 0]),
        ([ [1, 2], [3, 4], [5, 6] ], [3, 4]),
    ])
def test_daily_mean(test, expected):
    """Test mean function works for array of zeroes and positive integers."""
    from inflammation.models import daily_mean
    npt.assert_array_equal(daily_mean(np.array(test)), np.array(expected))


@pytest.mark.parametrize(
    "test, expected",
    [
        ([ [1, 2], [3, 4], [5, 6] ], [5, 6]),
        ([ [-1, 2], [-3, 10], [-5, 6] ], [-1, 10]),
    ])
def test_daily_max(test, expected):
    """Test mean function works for array of zeroes and positive integers."""
    from inflammation.models import daily_max
    npt.assert_array_equal(daily_max(np.array(test)), np.array(expected))


@pytest.mark.parametrize(
    "test, expected",
    [
        ([[0, 0, 0], [0, 0, 0], [0, 0, 0]], [0, 0, 0]),
        ([[4, 2, 5], [1, 6, 2], [4, 1, 9]], [1, 1, 2]),
        ([[4, -2, 5], [1, -6, 2], [-4, -1, 9]], [-4, -6, 2]),
    ])

def test_daily_min(test, expected):
    """Test mean function works for array of zeroes and positive integers."""
    from inflammation.models import daily_min
    npt.assert_array_equal(daily_min(np.array(test)), np.array(expected))

def test_daily_min_string():
    """Test for TypeError when passing strings"""
    from inflammation.models import daily_min

    with pytest.raises(TypeError):
        error_expected = daily_min([['Hello', 'there'], ['General', 'Kenobi']])

@pytest.mark.parametrize(
    "test, expected, expect_raises",
    [
        ([[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]], None),
        ([[1, 1, 1], [1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1], [1, 1, 1]], None),
        ([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[0.33, 0.67, 1], [0.67, 0.83, 1], [0.78, 0.89, 1]], None),
        (
            [[-1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[0, 0.67, 1], [0.67, 0.83, 1], [0.78, 0.89, 1]],
            ValueError,
        ),
        (
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[0.33, 0.67, 1], [0.67, 0.83, 1], [0.78, 0.89, 1]],
            None,
        ),
        (
            'hello',
            None,
            TypeError,
        ),
        (
            3,
            None,
            TypeError,
        )
    ])

def test_patient_normalise(test, expected, expect_raises):
    """Test normalisation works for arrays of one and positive integers."""
    from inflammation.models import patient_normalise
    if isinstance(test, list):
        test = np.array(test)
    if expect_raises is not None:
        with pytest.raises(expect_raises):
          result = patient_normalise(test)
          npt.assert_allclose(result, np.array(expected), rtol=1e-2, atol=1e-2)

    else:
        result = patient_normalise(test)
        npt.assert_allclose(result, np.array(expected), rtol=1e-2, atol=1e-2)

def test_compute_data_mock_source():
  from inflammation.compute_data import analyse_data
  data_source = Mock()
  data_source.load_inflammation_data.return_value = [[[0, 2, 0]],
                                                     [[0, 1, 0]]]

  result = analyse_data(data_source)
  npt.assert_array_almost_equal(result, [0, math.sqrt(0.25) ,0])


def test_analyse_data():
  from inflammation.compute_data import analyse_data, CSVDataSource
  path = Path.cwd() / "./data"
  data_source = CSVDataSource(path)
  result = analyse_data(data_source)
  expected_output = [0.        , 0.22510286, 0.18157299, 0.1264423 , 0.9495481 ,
       0.27118211, 0.25104719, 0.22330897, 0.89680503, 0.21573875,
       1.24235548, 0.63042094, 1.57511696, 2.18850242, 0.3729574 ,
       0.69395538, 2.52365162, 0.3179312 , 1.22850657, 1.63149639,
       2.45861227, 1.55556052, 2.8214853 , 0.92117578, 0.76176979,
       2.18346188, 0.55368435, 1.78441632, 0.26549221, 1.43938417,
       0.78959769, 0.64913879, 1.16078544, 0.42417995, 0.36019114,
       0.80801707, 0.50323031, 0.47574665, 0.45197398, 0.22070227]

  npt.assert_array_almost_equal(result, expected_output)

@pytest.mark.parametrize('data,expected_output', [
  ([[[0, 1, 0], [0, 2, 0]]], [0, 0, 0]),
  ([[[0, 2, 0]], [[0, 1, 0]]], [0, math.sqrt(0.25), 0]),
  ([[[0, 1, 0], [0, 2, 0]], [[0, 1, 0], [0, 2, 0]]], [0, 0, 0])
],
                       ids=['Two patients in same file', 'Two patients in different files',
                            'Two identical patients in two different files'])
def test_compute_standard_deviation_by_day(data, expected_output):
  from inflammation.compute_data import compute_standard_deviation_by_day

  result = compute_standard_deviation_by_day(data)
  npt.assert_array_almost_equal(result, expected_output)