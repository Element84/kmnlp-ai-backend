# type: ignore[reportUnknownMemberType]
# ruff: noqa: ERA001
import pytest

from tests.experiment.stats import f1_score, precision, recall


def test_precision_returns_none_if_none_predicted():
    predicted: set[int] = set()
    expected = {1}
    assert precision(predicted, expected) is None


def test_precision_return_1_when_completely_accurate():
    predicted = {1, 2, 3}
    expected = {1, 2, 3}

    assert precision(predicted, expected) == 1.0


def test_precision_returns_0_when_completely_false():
    predicted = {4}
    expected = {1, 2, 3}

    assert precision(predicted, expected) == 0.0


def test_presision_returns_fraction_of_predicted_that_are_correct():
    predicted = {1, 10, 11, 12}
    expected = {1, 2, 3, 4}

    assert precision(predicted, expected) == 0.25


def test_precision_returns_1_when_underlabeling():
    predicted = {1}
    expected = {1, 2, 3, 4}

    assert precision(predicted, expected) == 1.0


def test_precision_returns_expected_with_both_false_positives_and_negatives():
    predicted = {1, 2, 10, 11, 12}
    expected = {1, 2, 3, 4}

    assert precision(predicted, expected) == 0.4


def test_precision_works_on_strings():
    predicted = {"Call", "me", "Ishmael"}
    expected = {"Call", "me", "maybe", "Carly"}

    assert precision(predicted, expected) == pytest.approx(0.666666666666666667)


def test_recall_returns_1_if_none_expected():
    predicted: set[int] = set()
    expected: set[int] = set()

    assert recall(predicted, expected) == 1.0


def test_recall_returns_1_if_none_expected_when_some_predicted():
    predicted: set[int] = {1}
    expected: set[int] = set()

    assert recall(predicted, expected) == 1.0


def test_recall_returns_1_when_all_identified():
    predicted = {1, 2, 3, 4}
    expected = {1, 2, 3, 4}

    assert recall(predicted, expected) == 1.0


def test_recall_returns_0_when_all_false():
    predicted = {11, 12, 13, 14}
    expected = {1, 2, 3, 4}

    assert recall(predicted, expected) == 0.0


def test_recall_returns_0_when_overlabeling():
    predicted = {1, 2, 3, 4, 5}
    expected = {1, 2, 3, 4}

    assert recall(predicted, expected) == 1.0


def test_recall_returns_expected_when_both_false_positives_and_negatives():
    predicted = {1, 2, 10, 11, 12}
    expected = {1, 2, 3, 4}

    assert recall(predicted, expected) == 0.5


def test_recall_works_on_strings():
    predicted = {"Call", "me", "Ishmael"}
    expected = {"Call", "me", "maybe", "Carly"}

    assert recall(predicted, expected) == 0.5


def test_f1_returns_1_when_none_predicted_and_none_expected():
    predicted: set[int] = set()
    expected: set[int] = set()

    assert f1_score(predicted, expected) == 1.0


def test_f1_returns_1_when_none_predicted_but_something_expected():
    predicted: set[int] = set()
    expected = {1}

    # precision=None, recall=0.0
    assert f1_score(predicted, expected) == 0.0


def test_f1_returns_1_when_completely_accurate():
    predicted = {1, 2, 3}
    expected = {1, 2, 3}

    # precision=1.0, recall=1.0
    assert f1_score(predicted, expected) == 1.0


def test_f1_returns_0_when_completely_false():
    predicted = {4}
    expected = {1, 2, 3}

    assert f1_score(predicted, expected) == 0.0


def test_f1_returns_expected_value_when_precision_and_recall_equal():
    predicted = {1, 10, 11, 12}
    expected = {1, 2, 3, 4}

    # precision = 0.25, recall = 0.25
    assert f1_score(predicted, expected) == pytest.approx(0.25)


def test_f1_returns_expected_with_both_false_positives_and_negatives():
    predicted = {1, 2, 10, 11, 12}
    expected = {1, 2, 3, 4}

    # precision = 0.4, recall = 0.5
    # 2 * 0.4 * 0.5 / (0.4 + 0.5)
    # = 0.4 / (0.4 + 0.5)
    # = 0.4 / 0.9
    # = 4 / 9
    assert f1_score(predicted, expected) == pytest.approx(0.4444444444444444)


def test_f1_returns_expected_when_overlabeling():
    predicted = {1, 2, 3, 4, 5}
    expected = {1, 2, 3, 4}

    # precision = 0.8, recall = 1.0
    # 2 * 0.8 * 1.0 / (0.8 + 1.0)
    # = 1.6 / (1.8)
    # = 16 / 18
    # = 8 / 9
    assert f1_score(predicted, expected) == pytest.approx(0.8888888888888888)


def test_f1_returns_expected_when_underlabeling():
    predicted = {1}
    expected = {1, 2, 3, 4}

    # precision = 1.0, recall = 0.25
    assert f1_score(predicted, expected) == pytest.approx(0.4)


def test_f1_works_on_strings():
    predicted = {"Call", "me", "Ishmael"}
    expected = {"Call", "me", "maybe", "Carly"}

    # precision = 0.6666..., recall = 0.5
    assert f1_score(predicted, expected) == pytest.approx(0.5714285714285714)
