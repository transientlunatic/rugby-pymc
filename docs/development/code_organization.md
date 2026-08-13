# Code Organization & Testing Guide

## Recent Improvements (Phase 3 Housekeeping)

This document outlines the new structure for the rugby-ranking project.

## 1. Shared Utilities Module

All repeated patterns have been consolidated into `rugby_ranking.utils/`:

### `utils.logging`

Consistent formatting for CLI output:

```python
from rugby_ranking.utils.logging import (
    print_section, print_success, print_error, print_warning
)

print_section("DATA LOADING")
print_success("Data loaded successfully")
print_warning("Some data may be incomplete")
print_error("Failed to find checkpoint")
```

### `utils.cli_helpers`

Common CLI operations:

```python
from rugby_ranking.utils.cli_helpers import (
    load_checkpoint, setup_data, format_large_number
)

# Load trained model
model, trace = load_checkpoint("latest")

# Prepare data
dataset, df = setup_data(
    data_dir=Path("/path/to/Rugby-Data"),
    verbose=True
)

# Format numbers
print(f"Loaded {format_large_number(len(df))} records")
```

### `utils.constants`

Centralized constants for positions and scoring:

```python
from rugby_ranking.utils.constants import (
    KICKING_POSITIONS,
    TRY_SCORING_POSITIONS,
    SCORING_ADJUSTMENT,
    STARTERS,
)

# Use in validation
if position in KICKING_POSITIONS:
    # Check kicking scores
    pass
```

## 2. Notebook Utilities

Simplified notebook setup with `rugby_ranking.notebook_utils`:

### Before (Repeated in every notebook):

```python
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from rugby_ranking.model.data import MatchDataset

# Manual path handling
DATA_DIR = Path("../../Rugby-Data")
dataset = MatchDataset(DATA_DIR)
dataset.load_json_files()
df = dataset.to_dataframe(played_only=True)
df = df[df['position'].between(1, 23)].copy()
```

### After (Single line):

```python
from rugby_ranking.notebook_utils import setup_notebook_environment

dataset, df, model_dir = setup_notebook_environment()
```

### Additional utilities:

```python
from rugby_ranking.notebook_utils import (
    setup_notebook_environment,
    load_model_and_trace,
    print_summary,
    get_top_players,
    configure_plot_style,
    compare_seasons,
)

# Print formatted summary
print_summary(df, "My Analysis")

# Load model
model, trace = load_model_and_trace("latest")

# Get top players
tops = get_top_players(trace, model, score_type="tries", top=20)
```

## 3. Unit Tests

New comprehensive test suite in `tests/`:

### Running tests:

```bash
# All tests
pytest

# Specific test file
pytest tests/test_core_model.py

# With coverage report
pytest --cov=rugby_ranking --cov-report=html

# Specific marker
pytest -m "not slow"
```

### Test structure:

- `test_core_model.py`: Model building and configuration
- `test_data_loading.py`: Data loading and parsing
- `test_data_validation.py`: Data quality checks
- Integration tests in existing test files

### Writing tests:

```python
import pytest
import pandas as pd
from rugby_ranking.model.core import RugbyModel

class TestRugbyModel:
    @pytest.fixture
    def sample_data(self):
        """Provide test data."""
        return pd.DataFrame(...)
    
    def test_initialization(self):
        """Test model initialization."""
        model = RugbyModel()
        assert model.model is None  # Not built yet
```

## 4. Sphinx Documentation

Generate HTML documentation:

```bash
# Install docs dependencies
pip install -e ".[docs]"

# Build documentation
cd docs
sphinx-build -b html . _build

# View in browser
open _build/index.html  # macOS
xdg-open _build/index.html  # Linux
```

### Documentation structure:

```
docs/
├── conf.py                      # Sphinx configuration
├── index.md                     # Main landing page
├── getting_started/
│   ├── installation.md
│   ├── quickstart.md
│   └── architecture.md
├── guides/
│   ├── model_fundamentals.md
│   ├── weekly_workflow.md
│   ├── predictions.md
│   ├── squad_analysis.md
│   └── paths_to_victory.md
└── api/
    ├── model.md
    ├── data.md
    ├── inference.md
    ├── predictions.md
    └── utils.md
```

## 5. Dependency Management

### Development setup:

```bash
pip install -e ".[dev]"
```

### Includes:
- `pytest` & `pytest-cov` for testing
- `jupyter` for notebooks
- `black` & `ruff` for code formatting

### Documentation setup:

```bash
pip install -e ".[docs]"
```

### Includes:
- `sphinx` for documentation generation
- `pydata-sphinx-theme` for modern theme
- `myst-parser` for Markdown support

## 6. DRY Improvements

### Data Validation

Previously: Same logic repeated in `train_model.py`, `cli.py`, notebooks

Now: Centralized in `model/data_validation.py`:

```python
from rugby_ranking.model.data_validation import (
    detect_kicking_anomalies,
    clean_kicking_data,
    validate_position_scores,
)

anomalies = detect_kicking_anomalies(df)
df_clean = clean_kicking_data(df, strategy='remove')
```

### Error Handling

Previously: Different messages/formats throughout codebase

Now: Consistent via `utils.logging`:

```python
from rugby_ranking.utils.logging import print_success, print_error

try:
    model, trace = load_checkpoint(name)
    print_success("Model loaded")
except Exception as e:
    print_error(f"Failed: {e}")
    raise
```

## 7. Project Plan Reorganization

`PLAN.md` now clearly shows:

- **Phase 1 ✅**: Core model
- **Phase 2 ✅**: Model refinements
- **Phase 3 🔄**: Infrastructure & quality (current)
- **Phase 4 📋**: Match & tournament predictions (backlog)
- **Phase 5 📋**: Squad analysis (backlog)
- **Future**: Post-tournament improvements

Separated detailed implementation notes from high-level roadmap.

## 8. Next Steps

1. **Update notebooks** (Phase 3):
   - Replace boilerplate with `setup_notebook_environment()`
   - Use `notebook_utils` helpers throughout

2. **Run full test suite**:
   ```bash
   pytest --cov=rugby_ranking
   ```

3. **Validate documentation**:
   ```bash
   cd docs && sphinx-build -b html . _build
   ```

4. **Refactor CLI** to use new utils:
   - Replace print statements with `utils.logging`
   - Use `cli_helpers` for common operations

## Quick Reference

| Need | Use | Location |
|------|-----|----------|
| Formatting output | `print_section`, `print_success` | `utils.logging` |
| Load data | `setup_data` | `utils.cli_helpers` |
| Constants | `KICKING_POSITIONS`, etc | `utils.constants` |
| Notebook setup | `setup_notebook_environment` | `notebook_utils` |
| Model loading | `load_checkpoint` | `utils.cli_helpers` |
| Validation | `detect_kicking_anomalies` | `model.data_validation` |
| Testing | `pytest` | `tests/` |
