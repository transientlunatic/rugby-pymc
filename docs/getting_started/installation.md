# Installation

## Prerequisites

- Python 3.10 or higher
- Git

## Setup

### 1. Clone and Install

```bash
git clone https://github.com/transientlunatic/rugby-ranking.git
cd rugby-ranking
pip install -e .
```

### 2. Install Optional Dependencies

For full functionality including notebook support:

```bash
pip install -e ".[dev]"
```

### 3. Configure Data Source

Point to the [Rugby-Data](https://github.com/transientlunatic/Rugby-Data) repository:

```bash
# Set environment variable or pass --data-dir to CLI commands
export RUGBY_DATA_DIR=/path/to/Rugby-Data
```

## Verification

Verify the installation:

```bash
python -c "import rugby_ranking; print(rugby_ranking.__version__)"
rugby-ranking --help
```

## Next Steps

- Read the [Quick Start Guide](quickstart)
- Review [Model Fundamentals](../guides/model_fundamentals)
- Check out example [notebooks](../../notebooks)
