"""
Parameter validation utilities for StatLab statistical laws and tests.
"""
import numpy as np
from typing import Union
from i18n.translations import t

def validate_positive(x: float, label: str, lang: str = "en") -> float:
    """Validates that a numeric parameter is strictly positive (> 0)."""
    if x <= 0:
        raise ValueError(t("err_must_be_positive", lang).format(label=label, x=x))
    return x

def validate_non_negative(x: float, label: str, lang: str = "en") -> float:
    """Validates that a numeric parameter is non-negative (>= 0)."""
    if x < 0:
        raise ValueError(t("err_must_be_non_negative", lang).format(label=label, x=x))
    return x

def validate_range(x: float, lower: float, upper: float, label: str, lower_inclusive: bool = True, upper_inclusive: bool = True, lang: str = "en") -> float:
    """Validates that x is within [lower, upper] range."""
    valid_lower = x >= lower if lower_inclusive else x > lower
    valid_upper = x <= upper if upper_inclusive else x < upper
    if not (valid_lower and valid_upper):
        l_bracket = "[" if lower_inclusive else "("
        u_bracket = "]" if upper_inclusive else ")"
        raise ValueError(t("err_must_be_in_range", lang).format(
            label=label, l_bracket=l_bracket, lower=lower, upper=upper, u_bracket=u_bracket, x=x))
    return x

def validate_positive_integer(x: Union[int, float], label: str, lang: str = "en") -> int:
    """Validates that x is an integer and strictly positive (> 0)."""
    if not (isinstance(x, (int, float, np.integer)) and float(x).is_integer() and x > 0):
        raise ValueError(t("err_must_be_positive_integer", lang).format(label=label, x=x))
    return int(x)

def validate_non_negative_integer(x: Union[int, float], label: str, lang: str = "en") -> int:
    """Validates that x is an integer and non-negative (>= 0)."""
    if not (isinstance(x, (int, float, np.integer)) and float(x).is_integer() and x >= 0):
        raise ValueError(t("err_must_be_non_negative_integer", lang).format(label=label, x=x))
    return int(x)

def validate_probability(x: float, label: str, lang: str = "en") -> float:
    """Validates that x is a probability between 0 and 1 inclusive."""
    return validate_range(x, 0.0, 1.0, label, lower_inclusive=True, upper_inclusive=True, lang=lang)
