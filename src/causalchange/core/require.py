from __future__ import annotations

import importlib


def require_optional(
    import_name: str,
    *,
    package_name: str | None = None,
    extra: str | None = None,
    purpose: str | None = None,
):
    try:
        return importlib.import_module(import_name)
    except ImportError as exc:
        install_name = package_name or import_name
        if extra is not None:
            install_hint = f'pip install "causalchange[{extra}]"'
        else:
            install_hint = f"pip install {install_name}"

        msg = f"Optional dependency {install_name!r} is required"
        if purpose:
            msg += f" for {purpose}"
        msg += f". Install it with `{install_hint}`."

        raise ImportError(msg) from exc


def _require_matplotlib():
    plt = require_optional(
        "matplotlib.pyplot",
        package_name="matplotlib",
        extra="plotting",
        purpose="plotting",
    )
    return plt


def _require_rpt():
    rpt = require_optional(
        "ruptures",
        extra="spacetime",
        purpose="PELT changepoint detection",
    )
    return rpt


def _require_cit():
    cit_module = require_optional(
        "causallearn.utils.cit",
        package_name="causal-learn",
        extra="spacetime",
        purpose="KCI mechanism testing",
    )
    CIT = cit_module.CIT
    return CIT


def _require_hyppo():
    hyppo_mmd = require_optional(
        "hyppo.ksample",
        package_name="hyppo",
        extra="spacetime",
        purpose="MMD mechanism testing",
    )
    MMD = hyppo_mmd.MMD
    return MMD
