# Testing

## Running Tests

```bash
# Install test dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run with coverage
pytest --cov=rugby_ranking --cov-report=html

# Run specific test file
pytest tests/test_core_model.py

# Run specific test
pytest tests/test_core_model.py::TestRugbyModel::test_initialization

# Skip slow tests
pytest -m "not slow"
```

## Test Organization

- `tests/test_core_model.py`: Model construction and configuration
- `tests/test_data_loading.py`: Data loading and normalization
- `tests/test_data_validation.py`: Data quality checks
- Other files: Integration and component tests

## Writing Tests

### Setup

```python
import pytest
import pandas as pd
from rugby_ranking.model import RugbyModel

@pytest.fixture
def sample_data():
    """Provide test data."""
    return pd.DataFrame({
        'player_name': ['A', 'B', 'C'],
        'position': [10, 1, 15],
        ...
    })
```

### Structure

```python
class TestMyComponent:
    def test_basic_functionality(self, sample_data):
        """Test something specific."""
        result = my_function(sample_data)
        assert result is not None
    
    def test_error_handling(self):
        """Test error conditions."""
        with pytest.raises(ValueError):
            bad_function()
```

## Statistical Calibration Tests

A separate category of tests verifies that the inference engine itself is correctly calibrated, using Simulation-Based Calibration (SBC) — the same P-P test methodology used in gravitational wave astronomy. These tests are slow by design and are excluded from the normal test run.

```bash
# Run VI-based SBC (~15 min for the default 30 simulations)
pytest tests/test_statistical_calibration.py -m statistical -v -s

# Fewer simulations for a quick check
pytest tests/test_statistical_calibration.py -m statistical -v -s --sbc-sims 10

# MCMC reference (gold standard, several hours)
pytest tests/test_statistical_calibration.py -m "statistical and slow" -v -s
```

Each run saves a P-P plot to `tests/statistical_outputs/`.

**When to run:** Before and after major structural model changes (adding/removing parameters, changing the likelihood, switching inference method).

See [Model Validation](../guides/model_validation) for a full explanation of the SBC method and how to interpret results.

### Test Markers

| Marker | Description | Default run? |
|--------|-------------|-------------|
| `unit` | Fast, isolated unit tests | Yes |
| `integration` | Tests requiring real data | Yes |
| `slow` | Slow tests (MCMC-based, etc.) | No (`-m "not slow"`) |
| `statistical` | SBC / P-P calibration tests | No (`-m statistical`) |

## Coverage

Current coverage focuses on:
- Core model building (✓)
- Data loading (✓)
- Data validation (✓)
- Statistical calibration (✓ — `tests/test_statistical_calibration.py`)
- Predictions (partial)
- CLI commands (partial)

### Improving coverage

1. Identify uncovered lines: `coverage report --omit=tests`
2. Add tests for critical paths
3. Test error conditions and edge cases
