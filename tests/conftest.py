"""
Shared pytest configuration and fixtures.

pytest_addoption hooks MUST live here (in conftest.py), not in test files.
"""


def pytest_addoption(parser):
    parser.addoption(
        "--sbc-sims",
        type=int,
        default=30,
        help="Number of SBC simulations for statistical calibration tests (default: 30)",
    )
