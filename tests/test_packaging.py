"""What the wheel promises about itself."""

from importlib.metadata import PackageNotFoundError, requires

import pytest


def test_the_package_declares_no_required_runtime_dependencies():
    """"Zero runtime dependencies" is a claim about installed metadata."""
    try:
        declared = requires("gpu-quiescence") or []
    except PackageNotFoundError:  # pragma: no cover - source checkout, not an install
        pytest.skip("run against an installed gpu-quiescence")
    required = [r for r in declared if "extra ==" not in r]
    assert required == []
    # ...and everything else is opt-in, by name.
    assert {"psutil", "torch"} <= {r.split(";")[0].strip() for r in declared}
