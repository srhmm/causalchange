### Installation

---

**Dev install**
```bash
git clone https://github.com/srhmm/causalchange.git
cd causalchange
conda create -n causalchange python=3.10 -y
conda activate causalchange
pip install -e ".[dev,spacetime,notebooks]"
```

**Format** Run linting and formatting with

```bash
ruff check .
ruff format .
```

Before committing, the repository uses pre-commit hooks.

```bash
pre-commit run --all-files
```

**Tests**
Run the full test suite with `pytest`.
