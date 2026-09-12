# Contributing

Contributions that improve correctness, reproducibility, or documentation are welcome.

## Development setup

```bash
conda env create -f environment.yml
conda activate amo
pip install -r requirements/requirements_dev.txt
pre-commit install
```

Before opening a pull request, run:

```bash
black --check amo scripts tests
ruff check amo scripts tests
pytest -q
```

Please keep algorithmic changes separate from refactors and add a regression test for every change to the update equations or checkpoint format.
