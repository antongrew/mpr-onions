"""pytest wrapper around the package's own self-test suite.

The real checks live in `format/selftest.py` so the suite stays runnable with
nothing installed.  This file just exposes them to pytest and runs them under
several parameter sets, which is where most structural bugs have surfaced.
"""

import importlib

import pytest

from format.params import Params

PARAM_SETS = [
    pytest.param(Params(), id="d5-nu5-p4"),
    pytest.param(Params(d=3, nu=3, alpha=0.67, p=2), id="d3-nu3-p2"),
    pytest.param(Params(d=7, nu=7, alpha=0.5, p=3), id="d7-nu7-p3"),
]


@pytest.mark.parametrize("params", PARAM_SETS)
def test_selftest_suite(params):
    # selftest keeps module-level counters, so reload it for each parameter set.
    selftest = importlib.reload(importlib.import_module("format.selftest"))
    assert selftest.main(params) == 0, "see captured output for the failing check"


def test_parameters_validate():
    for p in (s.values[0] for s in PARAM_SETS):
        p.validate()
        assert p.onion_len == (
            p.d * p.tau_K + p.d * p.tau_H + p.tau_M + p.tau_U
        )
