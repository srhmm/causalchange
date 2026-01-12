import os
import sys
import types


def _install_pygam_stub() -> None:
    # The package imports `pygam.GAM` unconditionally, but GAM scoring is optional.
    # Provide a minimal stub so core functionality can be tested without pulling pygam into CI.
    if "pygam" in sys.modules:
        return
    mod = types.ModuleType("pygam")

    class GAM:  # noqa: N801
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, X, y):
            return self

        def predict(self, X):
            # basic baseline prediction: zeros
            import numpy as np

            return np.zeros(len(X), dtype=float)

    mod.GAM = GAM
    sys.modules["pygam"] = mod


def pytest_configure():
    _install_pygam_stub()
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
