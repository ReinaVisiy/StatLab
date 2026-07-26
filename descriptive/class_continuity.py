"""
Class Continuity Helper Module.
Adjusts non-adjacent class boundaries into contiguous intervals.
"""
from typing import List, Tuple

def continuize_classes(classes: List[Tuple[float, float]]) -> Tuple[List[Tuple[float, float]], float, bool]:
    """
    Adjusts non-adjacent continuous class bounds [a_i, b_i] into contiguous bounds.
    If b_i < a_{i+1}, computes epsilon = (a_{i+1} - b_i)/2 and expands each class:
    [a_i - epsilon, b_i + epsilon].
    Returns (adjusted_classes, epsilon, was_adjusted).
    """
    if len(classes) <= 1:
        return classes, 0.0, False

    # Sort classes by lower bound
    sorted_classes = sorted(classes, key=lambda x: x[0])
    
    # Check if gap exists
    gaps = []
    for i in range(len(sorted_classes) - 1):
        gap = sorted_classes[i+1][0] - sorted_classes[i][1]
        if gap > 0:
            gaps.append(gap)

    if not gaps:
        return sorted_classes, 0.0, False

    # Use average or constant gap
    epsilon = sum(gaps) / (2.0 * len(gaps))
    
    adjusted = []
    for i, (lower, upper) in enumerate(sorted_classes):
        new_lower = lower - epsilon
        new_upper = upper + epsilon
        adjusted.append((new_lower, new_upper))

    return adjusted, epsilon, True
