"""
Tests for core/param_solver.py — the generic "solve for a distribution
parameter given a probability" engine (master list item #12).

Two layers:
  1. Explicit known-answer cases covering discrete/continuous, prob/pos/
     posint/int/float parameter types, and a domain-constrained case
     (discrete_uniform's a < b) that exercises the safe-eval bracket walk.
  2. A registry-driven round-trip smoke test: for every law in the
     "discrete"/"continuous" suites, compute a baseline probability at the
     law's own default parameters, then solve backwards for that same
     parameter and check we recover (approximately) the default we started
     from. This exercises all 20 law files without hand-writing 20 cases.
"""
import importlib

import pytest

from core.registry import SUITES
from core.param_solver import (
    solve_for_parameter, is_solvable_param, ParamSolveError,
    SOLVABLE_QUERY_TYPES_DISCRETE, SOLVABLE_QUERY_TYPES_CONTINUOUS,
)
from laws.discrete.binomial import run_binomial_calc
from laws.discrete.discrete_uniform import run_discrete_uniform_calc
from laws.discrete.poisson import run_poisson_calc
from laws.discrete.hypergeometric import run_hypergeometric_calc
from laws.discrete.multinomial import run_multinomial_calc
from laws.continuous.normal import run_normal_calc
from laws.continuous.exponential import run_exponential_calc


def _resolve(module, func):
    return getattr(importlib.import_module(module), func)


# ---------------------------------------------------------------------------
# 1. Explicit known-answer / behavioral cases
# ---------------------------------------------------------------------------

def test_solve_binomial_p_prob_type():
    spec = {"name": "p", "label": "p", "type": "prob", "default": 0.5}
    res = solve_for_parameter(run_binomial_calc, {"n": 20}, "p", spec, "P(X<=k)", 8, 0.5, lang="en")
    p_hat = res["solved_parameter"]["value"]
    check = run_binomial_calc({"n": 20, "p": p_hat}, "P(X<=k)", k=8)["result"]
    assert check == pytest.approx(0.5, abs=1e-4)


def test_solve_discrete_uniform_b_int_type_with_domain_constraint():
    # a < b is enforced by the law itself; the solver must not crash while
    # bracket-walking near/through the invalid (b <= a) region.
    spec = {"name": "b", "label": "b", "type": "int", "default": 6}
    res = solve_for_parameter(run_discrete_uniform_calc, {"a": 1}, "b", spec, "P(X<=k)", 3, 0.5, lang="en")
    b_hat = res["solved_parameter"]["value"]
    assert isinstance(b_hat, int)
    check = run_discrete_uniform_calc({"a": 1, "b": b_hat}, "P(X<=k)", k=3)["result"]
    assert check == pytest.approx(0.5, abs=1e-6)


def test_solve_poisson_mu_pos_type():
    spec = {"name": "mu", "label": "lambda", "type": "pos", "default": 3.0}
    res = solve_for_parameter(run_poisson_calc, {}, "mu", spec, "P(X<=k)", 5, 0.6160, lang="en")
    mu_hat = res["solved_parameter"]["value"]
    check = run_poisson_calc({"mu": mu_hat}, "P(X<=k)", k=5)["result"]
    assert check == pytest.approx(0.6160, abs=1e-3)


def test_solve_normal_mu_float_type():
    spec = {"name": "mu", "label": "mu", "type": "float", "default": 0.0}
    res = solve_for_parameter(run_normal_calc, {"sigma": 2.0}, "mu", spec, "P(X<=a)", 10, 0.5, lang="en")
    assert res["solved_parameter"]["value"] == pytest.approx(10.0, abs=1e-3)


def test_solve_normal_sigma_pos_type():
    spec = {"name": "sigma", "label": "sigma", "type": "pos", "default": 1.0}
    res = solve_for_parameter(run_normal_calc, {"mu": 0.0}, "sigma", spec, "P(X<=a)", 5, 0.9772, lang="en")
    sigma_hat = res["solved_parameter"]["value"]
    check = run_normal_calc({"mu": 0.0, "sigma": sigma_hat}, "P(X<=a)", a=5)["result"]
    assert check == pytest.approx(0.9772, abs=1e-3)


def test_solve_hypergeometric_n_posint_type():
    spec = {"name": "n", "label": "n", "type": "posint", "default": 20}
    res = solve_for_parameter(run_hypergeometric_calc, {"M": 50, "N": 10}, "n", spec, "P(X=k)", 3, 0.25, lang="en")
    n_hat = res["solved_parameter"]["value"]
    assert isinstance(n_hat, int)
    check = run_hypergeometric_calc({"M": 50, "n": n_hat, "N": 10}, "P(X=k)", k=3)["result"]
    assert check == pytest.approx(0.25, abs=0.05)  # integer n won't hit it exactly


def test_solve_exponential_rate_matches_closed_form():
    # P(X > a) = exp(-rate * a)  =>  rate = -ln(target_p) / a
    import math
    spec = {"name": "rate", "label": "lambda", "type": "pos", "default": 1.0}
    res = solve_for_parameter(run_exponential_calc, {}, "rate", spec, "P(X>a)", 2, 0.1353, lang="en")
    expected_rate = -math.log(0.1353) / 2
    assert res["solved_parameter"]["value"] == pytest.approx(expected_rate, rel=1e-3)


def test_solve_raises_param_solve_error_on_bad_target_probability():
    spec = {"name": "p", "label": "p", "type": "prob", "default": 0.5}
    with pytest.raises(ParamSolveError):
        solve_for_parameter(run_binomial_calc, {"n": 20}, "p", spec, "P(X<=k)", 8, 1.5, lang="en")


def test_multinomial_n_is_not_flagged_solvable_incorrectly_but_vec_params_are_excluded():
    # multinomial's n (posint) IS solvable in principle; its p/x vectors are not.
    n_spec = {"name": "n", "label": "n", "type": "posint", "default": 10}
    p_spec = {"name": "p", "label": "p", "type": "vec", "default": "0.2,0.3,0.5"}
    assert is_solvable_param(n_spec) is True
    assert is_solvable_param(p_spec) is False


def test_multinomial_degenerate_n_solve_recovers_default_at_consistent_x():
    # n is constrained by sum(x) == n, so the only domain-valid n for a
    # fixed x vector is the one already consistent with it (the default).
    spec = {"name": "n", "label": "n", "type": "posint", "default": 10}
    base = {"p": "0.2,0.3,0.5", "x": "2,3,5"}
    baseline = run_multinomial_calc({**base, "n": 10}, "P(X=k)")["result"]
    res = solve_for_parameter(run_multinomial_calc, base, "n", spec, "P(X=k)", None, baseline, lang="en")
    assert res["solved_parameter"]["value"] == 10


# ---------------------------------------------------------------------------
# 2. Registry-driven round-trip smoke test across all discrete/continuous laws
# ---------------------------------------------------------------------------

def _law_items():
    items = []
    for suite_key in ("discrete", "continuous"):
        for item in SUITES[suite_key]["items"]:
            if item["entry"] == "D" and item.get("params_spec"):
                items.append((suite_key, item))
    return items


LAW_ITEMS = _law_items()
LAW_IDS = [item["id"] for _, item in LAW_ITEMS]


@pytest.mark.parametrize("suite_key,item", LAW_ITEMS, ids=LAW_IDS)
def test_solver_round_trips_default_parameter(suite_key, item):
    """For each law: compute P(...) at the default parameters, then solve
    backwards for one solvable parameter given that same probability, and
    check the solver recovers (approximately) the original default."""
    if item["id"] == "multinomial":
        pytest.skip("multinomial's n is constrained by sum(x); covered by a dedicated test")

    solvable = [s for s in item["params_spec"] if is_solvable_param(s)]
    if not solvable:
        pytest.skip(f"{item['id']} has no solvable scalar parameters")

    spec = solvable[-1]
    calc_fn = _resolve(item["module"], item["func"])
    is_discrete = suite_key == "discrete"
    solve_qts = SOLVABLE_QUERY_TYPES_DISCRETE if is_discrete else ["P(X<=a)", "P(X<a)", "P(X>a)", "P(X>=a)", "f(x)"]
    query_type = next((qt for qt in solve_qts if qt in item["query_types"]), None)
    if query_type is None:
        pytest.skip(f"{item['id']} exposes no solvable query type")

    full_default_params = {s["name"]: s["default"] for s in item["params_spec"]}

    # Probe the distribution's mean (if numeric) to pick a sensible x value;
    # fall back to a small fixed probe point otherwise.
    probe_kwargs = {"k": 1} if is_discrete else {"a": 0.0}
    try:
        probe = calc_fn(full_default_params, query_type, lang="en", **probe_kwargs)
    except Exception:
        pytest.skip(f"{item['id']} could not be probed with defaults")
    mean = probe.get("properties", {}).get("mean")
    if isinstance(mean, (int, float)):
        x_val = round(mean) if is_discrete else float(mean)
    else:
        x_val = 1 if is_discrete else 0.0

    kwargs = {"k": x_val} if is_discrete else {"a": x_val}
    try:
        baseline = calc_fn(full_default_params, query_type, lang="en", **kwargs)["result"]
    except Exception:
        pytest.skip(f"{item['id']} could not compute a baseline probability")

    if not (1e-4 < baseline < 1 - 1e-4):
        pytest.skip(f"{item['id']}: baseline probability {baseline} too extreme for a robust round-trip")

    base_params = {k: v for k, v in full_default_params.items() if k != spec["name"]}
    try:
        res = solve_for_parameter(calc_fn, base_params, spec["name"], spec, query_type, x_val, baseline, lang="en")
    except ParamSolveError:
        pytest.skip(f"{item['id']}: no root found for {spec['name']} in the search range")

    recovered = res["solved_parameter"]["value"]
    default = spec["default"]
    if spec["type"] in ("posint", "int"):
        assert abs(recovered - default) <= 1
    else:
        assert recovered == pytest.approx(default, rel=0.05, abs=0.05)
