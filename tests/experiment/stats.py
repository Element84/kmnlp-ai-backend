def precision[T](predicted: set[T], expected: set[T]) -> float | None:
    """Calculate precision of a predicted set.

    precision: true positives / (true positives + false positives)

    Args:
        predicted (set[T]): The set of items a model predicted to be in a given
            set.
        expected (set[T]): The set of items known to be in the given set.

    Returns:
        The calculated precision of the set, or `None` if there are no items in
        the predicted set.
    """
    # If there are no predicted results, they aren't really accurate or
    # inaccurate. Return `None` and exclude from final result.
    if len(predicted) == 0:
        return None

    true_positives = len(predicted & expected)
    return true_positives / len(predicted)


def recall[T](predicted: set[T], expected: set[T]) -> float:
    """Calculate recall of a predicted set.

    recall: true positives / total actual positives (i.e. true positives + false negatives)

    Args:
        predicted (set[T]): The set of items a model predicted to be in a given
            set.
        expected (set[T]): The set of items known to be in the given set.

    Returns:
        The calculated recall of the set. If the expected set is empty, return
        1.0, indicating that all of the expected datasets are present.
    """
    if len(expected) == 0:
        return 1.0

    true_positives = len(predicted & expected)
    return true_positives / len(expected)


def f1_score[T](predicted: set[T], expected: set[T]) -> float:
    """Calculate the F1 score of a predicted set.

    f1 score:
        2 true positives / (2 * true positives + false positives + false negatives) ==
        2 * precision * recall / (precision + recall) ==
        2 / (recall^-1 + precision^-1)

    Args:
        predicted (set[T]): The set of items a model predicted to be in a given
            set.
        expected (set[T]): The set of items known to be in the given set.

    Returns:
        The calculated F1 score of the predicted set.
    """
    precision_score = precision(predicted, expected)
    recall_score = recall(predicted, expected)

    if precision_score is None:
        # Precision is None, only recall contributes to score.
        return recall_score

    if precision_score + recall_score == 0:
        # Avoid divide by zero:
        # Precision and recall are both 0, meaning F1 score should also be 0.
        return 0

    return 2 * precision_score * recall_score / (precision_score + recall_score)
