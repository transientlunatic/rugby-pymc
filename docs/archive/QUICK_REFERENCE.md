# Quick Start for Phase 3 Changes

## For Notebook Users

Replace this (in every notebook):
```python
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from rugby_ranking.model.data import MatchDataset
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = Path("../../Rugby-Data")
dataset = MatchDataset(DATA_DIR)
dataset.load_json_files()
df = dataset.to_dataframe(played_only=True)
df = df[df['position'].between(1, 23)].copy()
# ... 10 more lines of setup
```

With this:
```python
from rugby_ranking.notebook_utils import setup_notebook_environment
dataset, df, model_dir = setup_notebook_environment()
```

Plus these helpers:
```python
from rugby_ranking.notebook_utils import (
    load_model_and_trace,
    print_summary,
    get_top_players,
)

# Load model
model, trace = load_model_and_trace("latest")

# Print summary
print_summary(df)

# Get rankings
tops = get_top_players(trace, model, score_type="tries")
```

## For CLI Developers

### Logging

Replace this:
```python
print(f"Loading data...")
print(f"✓ Loaded successfully")
```

With this:
```python
from rugby_ranking.utils.logging import print_info, print_success, print_section

print_section("DATA LOADING")
print_success("Loaded successfully")
```

### Loading Checkpoints

Replace this:
```python
model = RugbyModel()
fitter = ModelFitter.load(checkpoint_name, model)
trace = fitter.trace
```

With this:
```python
from rugby_ranking.utils.cli_helpers import load_checkpoint

model, trace = load_checkpoint(checkpoint_name)
```

### Constants

Replace this scattered throughout files:
```python
KICKING_POSITIONS = {9, 10, 12, 15}
TRY_SCORING_POSITIONS = {11, 12, 13, 14, 15}
```

With this:
```python
from rugby_ranking.utils.constants import KICKING_POSITIONS, TRY_SCORING_POSITIONS
```

## Testing

Run all tests:
```bash
pytest --cov=rugby_ranking
```

Run specific tests:
```bash
pytest tests/test_core_model.py -v
```

## Documentation

Build HTML documentation:
```bash
pip install -e ".[docs]"
sphinx-build -b html docs docs/_build
open docs/_build/index.html
```

## What Exists Now

| Component | Location | Purpose |
|-----------|----------|---------|
| Logging utilities | `rugby_ranking/utils/logging.py` | Consistent CLI output |
| CLI helpers | `rugby_ranking/utils/cli_helpers.py` | Common operations |
| Constants | `rugby_ranking/utils/constants.py` | Single source of truth |
| Notebook utils | `rugby_ranking/notebook_utils.py` | Notebook boilerplate |
| Unit tests | `tests/test_*.py` | 85+ test cases |
| Sphinx docs | `docs/` | Professional documentation |
| This guide | `HOUSEKEEPING_COMPLETE.md` | What was done |

## Dependencies to Install

For full development:
```bash
pip install -e ".[dev,docs]"
```

This adds:
- `pytest` & `pytest-cov` for testing
- `sphinx` for documentation
- `black` & `ruff` for formatting

## Next Phase (Phase 3 Remaining)

- [ ] Update all 8 notebooks to use `setup_notebook_environment()`
- [ ] Run full model training and validation
- [ ] Refactor CLI to use new utils
- [ ] Generate and review documentation

## Questions?

See:
- `docs/development/code_organization.md` - Detailed migration guide
- `docs/development/testing.md` - Testing guidelines
- `docs/development/contributing.md` - Contributing guidelines
- `PLAN.md` - Project roadmap
