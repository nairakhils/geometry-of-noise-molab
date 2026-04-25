import numpy as np
import pytest

from src.schedules import a_of_t, adot_of_t, b_of_t, bdot_of_t


@pytest.mark.parametrize("t", [0.05, 0.25, 0.5, 0.75, 0.95])
def test_adot_matches_central_difference(t):
    h = 1e-6
    cd = (a_of_t(t + h) - a_of_t(t - h)) / (2 * h)
    assert np.isclose(float(adot_of_t(t)), float(cd), atol=1e-7)


@pytest.mark.parametrize("t", [0.05, 0.25, 0.5, 0.75, 0.95])
def test_bdot_matches_central_difference(t):
    h = 1e-6
    cd = (b_of_t(t + h) - b_of_t(t - h)) / (2 * h)
    assert np.isclose(float(bdot_of_t(t)), float(cd), atol=1e-7)


def test_adot_bdot_vectorized():
    t = np.linspace(0.05, 0.95, 19)
    h = 1e-6
    cd_a = (a_of_t(t + h) - a_of_t(t - h)) / (2 * h)
    cd_b = (b_of_t(t + h) - b_of_t(t - h)) / (2 * h)
    assert np.allclose(adot_of_t(t), cd_a, atol=1e-7)
    assert np.allclose(bdot_of_t(t), cd_b, atol=1e-7)
