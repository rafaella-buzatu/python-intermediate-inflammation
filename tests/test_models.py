"""Tests for statistics functions within the Model layer."""

import numpy as np
import numpy.testing as npt
import math
from unittest.mock import Mock
from pathlib import Path
import pytest

def test_daily_mean_zeros():
    """Test that mean function works for an array of zeros."""
    from inflammation.models import daily_mean

    test_input = np.array([[0, 0],
                           [0, 0],
                           [0, 0]])
    test_result = np.array([0, 0])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(test_input), test_result)


def test_daily_mean_integers():
    """Test that mean function works for an array of positive integers."""
    from inflammation.models import daily_mean

    test_input = np.array([[1, 2],
                           [3, 4],
                           [5, 6]])
    test_result = np.array([3, 4])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(test_input), test_result)

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