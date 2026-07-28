"""
Generic "solve for a distribution parameter given a probability" engine.

Rather than hand-writing bespoke root-finding logic in each of the 20
laws/*.py files, this module treats every law's existing run_<law>_calc()
function as a black-box probability engine and numerically inverts it:
given a fixed x/k value and a target probability p, it searches for the
value of one chosen parameter that makes calc_fn(params, query_type, ...)
return p. This keeps the root-finding logic implemented in exactly one
place (per the project's zero-duplicate-logic rule) instead of once per law.

The search is domain-safe: laws routinely reject some parameter values
(e.g. discrete_uniform requires a < b), so every trial evaluation is
wrapped and treated as "outside the domain" rather than crashing the
search, and the walk starts from the parameter's own default value
outward in both directions until it finds a domain boundary or a sign
change in (probability - target).

Exports: solve_for_parameter, is_solvable_param,
         SOLVABLE_QUERY_TYPES_DISCRETE, SOLVABLE_QUERY_TYPES_CONTINUOUS
"""
from typing import Callable, Optional
from i18n.translations import t as tt

# Query types that resolve to a single probability value and are therefore
# invertible for a parameter. "P(a<=X<=b)" (two unknowns) and "inverse"
# (already solves for x, not a parameter) are intentionally excluded.
SOLVABLE_QUERY_TYPES_DISCRETE = ["P(X=k)", "P(X<=k)", "P(X<k)", "P(X>k)", "P(X>=k)"]
SOLVABLE_QUERY_TYPES_CONTINUOUS = ["f(x)", "P(X<=a)", "P(X<a)", "P(X>a)", "P(X>=a)"]

# Parameter types this generic solver can search over. "vec" params (e.g.
# multinomial's category-probability/count vectors) are not solvable.
_SOLVABLE_PARAM_TYPES = {"prob", "pos", "posint", "int", "float"}


class ParamSolveError(ValueError):
    pass


def is_solvable_param(spec: dict) -> bool:
    return spec.get("type", "float") in _SOLVABLE_PARAM_TYPES


def _uses_k(query_type: str) -> bool:
    return query_type in SOLVABLE_QUERY_TYPES_DISCRETE


def _safe_eval(f, theta) -> Optional[float]:
    try:
        val = f(theta)
        if val is None:
            return None
        return float(val)
    except Exception:
        return None


def _find_valid_start(f, center):
    """Returns (point, f_value) for the first domain-valid point at or
    near `center`, nudging outward a few steps if center itself is
    rejected by the law's own validation."""
    val = _safe_eval(f, center)
    if val is not None:
        return center, val
    for delta in (1, 2, 3, 5, 8, 13):
        for cand in (center + delta, center - delta):
            val = _safe_eval(f, cand)
            if val is not None:
                return cand, val
    return None, None


def _bisect_between(f, lo, hi, f_lo, f_hi, is_int: bool, tol: float = 1e-9, max_iter: int = 200):
    """Standard bisection between two domain-valid points of opposite sign."""
    if f_lo == 0:
        return lo
    if f_hi == 0:
        return hi
    for _ in range(max_iter):
        if is_int:
            if hi - lo <= 1:
                return lo if abs(f_lo) <= abs(f_hi) else hi
            mid = (lo + hi) // 2
        else:
            mid = (lo + hi) / 2.0
            if (hi - lo) < tol:
                return mid
        f_mid = _safe_eval(f, mid)
        if f_mid is None:
            # midpoint outside domain (can happen near a boundary) —
            # nudge toward the still-valid side and keep going.
            mid = mid + (1 if is_int else tol * 10) if abs(hi - mid) < abs(mid - lo) else mid - (1 if is_int else tol * 10)
            f_mid = _safe_eval(f, mid)
            if f_mid is None:
                return lo if abs(f_lo) <= abs(f_hi) else hi
        if f_mid == 0:
            return mid
        if (f_lo < 0) == (f_mid < 0):
            lo, f_lo = mid, f_mid
        else:
            hi, f_hi = mid, f_mid
    return lo if abs(f_lo) <= abs(f_hi) else hi


def _bracket_search(f, center, is_int: bool, max_steps: int = 60):
    """Walks outward from `center` in both directions (each step growing)
    looking for a sign change in f, stopping a direction once it leaves
    the law's valid domain. Returns a root, or None if none is found."""
    start, f_start = _find_valid_start(f, center)
    if start is None:
        return None
    if f_start == 0:
        return start

    lo = hi = start
    f_lo = f_hi = f_start
    lo_blocked = hi_blocked = False
    step = 1 if is_int else max(abs(center) * 0.1, 0.1)

    for _ in range(max_steps):
        if not hi_blocked:
            cand = hi + step
            fc = _safe_eval(f, cand)
            if fc is None:
                hi_blocked = True
            elif fc == 0:
                return cand
            elif (f_hi < 0) != (fc < 0):
                return _bisect_between(f, hi, cand, f_hi, fc, is_int)
            else:
                hi, f_hi = cand, fc
        if not lo_blocked:
            cand = lo - step
            fc = _safe_eval(f, cand)
            if fc is None:
                lo_blocked = True
            elif fc == 0:
                return cand
            elif (f_lo < 0) != (fc < 0):
                return _bisect_between(f, cand, lo, fc, f_lo, is_int)
            else:
                lo, f_lo = cand, fc
        if lo_blocked and hi_blocked:
            break
        step = (step + 1) if is_int else step * 1.6

    return None


def solve_for_parameter(calc_fn: Callable, base_params: dict, solve_for: str,
                         param_spec: dict, query_type: str, x_val: float,
                         target_p: float, lang: str = "en") -> dict:
    """
    Finds the value of base_params[solve_for] such that the law's own
    calc_fn(params, query_type, k=x_val or a=x_val) returns target_p.

    Returns a result dict shaped exactly like a normal run_<law>_calc()
    result (steps/result/formula_latex/properties/plot_data), so the
    existing "D" entry results page can render it with no special-casing,
    plus a "solved_parameter" key with the solved name/value/label.
    """
    if query_type != "f(x)" and not (0.0 < target_p < 1.0):
        raise ParamSolveError(tt("err_must_be_in_range", lang).format(
            label="target probability p", l_bracket="(", lower=0, upper=1, u_bracket=")", x=target_p))

    ptype = param_spec.get("type", "float")
    is_int = ptype in ("posint", "int")
    center = param_spec.get("default", 1 if is_int else 1.0)

    def f(theta):
        trial = dict(base_params)
        trial[solve_for] = int(round(theta)) if is_int else theta
        kwargs = {"k": x_val} if _uses_k(query_type) else {"a": x_val}
        return float(calc_fn(trial, query_type, lang=lang, **kwargs)["result"]) - target_p

    root = _bracket_search(f, center, is_int)
    if root is None:
        raise ParamSolveError(tt("solve_no_root_error", lang).format(param=param_spec["label"]))
    if is_int:
        root = int(round(root))

    final_params = dict(base_params)
    final_params[solve_for] = root
    kwargs = {"k": x_val} if _uses_k(query_type) else {"a": x_val}
    final_result = calc_fn(final_params, query_type, lang=lang, **kwargs)

    check_val = float(final_result["result"])
    value_str = str(root) if is_int else f"{root:.6g}"

    solve_steps = [
        tt("solve_result_line", lang).format(eq=query_type, param=solve_for, value=value_str),
        tt("solve_verify_line", lang).format(param=solve_for, value=value_str, eq=query_type, check=f"{check_val:.6f}"),
    ]
    final_result["steps"] = solve_steps + list(final_result.get("steps", []))
    final_result["solved_parameter"] = {
        "name": solve_for,
        "label": param_spec["label"],
        "value": root,
        "value_str": value_str,
        "target_probability": target_p,
        "query_type": query_type,
        "x_val": x_val,
    }
    final_result["result"] = root
    return final_result
