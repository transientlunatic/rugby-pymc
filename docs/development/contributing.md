# Contributing

## Code Style

We follow PEP 8 with some project-specific conventions:

```bash
# Format code
black rugby_ranking tests

# Check linting
ruff check rugby_ranking tests

# Fix auto-fixable issues
ruff check --fix rugby_ranking tests
```

Configuration in `pyproject.toml`:
- Line length: 100 characters
- Linter: ruff (E, F, I, W)
- Import sorting: isort

## Pull Request Checklist

Before submitting a PR:

- [ ] Code is formatted with `black`
- [ ] Tests pass: `pytest`
- [ ] Coverage maintained or improved
- [ ] Documentation updated
- [ ] Commit messages are clear

## Setting Up Development Environment

```bash
# Clone repository
git clone https://github.com/transientlunatic/rugby-ranking
cd rugby-ranking

# Create virtual environment
python3.10 -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install in development mode
pip install -e ".[dev,docs]"

# Install pre-commit hooks (optional)
pip install pre-commit
pre-commit install
```

## Working with Notebooks

Notebooks should:
- Use `setup_notebook_environment()` for boilerplate
- Include explanatory markdown cells
- Be reproducible (all paths relative)
- Have clear section headings
- Save outputs (not large data files)

## Documentation

- Update `PLAN.md` for roadmap changes
- Add docstrings to all functions/classes
- Update relevant `.md` files in `docs/`
- Build locally: `sphinx-build -b html docs docs/_build`

## Questions?

- Check existing issues/PRs
- Review API documentation
- See [code_organization.md](code_organization.md) for module structure
