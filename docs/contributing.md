# Contributing

Contributions are welcome: bug reports, documentation fixes, new examples and
new features. This page describes how to set up a development environment, how
the checks run, and how a release is made.

## Development setup

Clone the repository and install it in editable mode with every extra, the
test tools and the documentation tools:

```bash
git clone https://github.com/bpogroup/gympn.git
cd gympn
pip install -e ".[dev]"
```

`requirements.txt` contains exactly this install line, so
`pip install -r requirements.txt` is equivalent.

PyTorch publishes CPU-only and CUDA-specific wheels on its own index. If you
want a specific build, install `torch` first by following
<https://pytorch.org/get-started/locally/> and then run the command above.

## Running the checks

The same three checks run in CI (`.github/workflows/ci.yml`) on every push and
pull request, on Python 3.10 to 3.13:

```bash
ruff check gympn tests      # errors and likely bugs only; style is not enforced
pytest                      # the test suite, about half a minute
mkdocs build --strict       # the documentation; broken links fail the build
```

`tests/test_packaging.py` checks the packaging itself: every third-party
import is declared in `pyproject.toml`, `import gympn` does not load an
optional dependency, training works with the extras missing, and the wheel
contains only the library. The tests that build the wheel are marked `slow`
and can be skipped with `pytest -m "not slow"`. The test that installs the
wheel into a fresh virtual environment downloads PyTorch, so it runs only when
`GYMPN_INSTALL_TEST=1` is set.

CI also builds the wheel and installs it without extras in a clean
environment, then trains and tests from outside the checkout, so the installed
package is exercised rather than the source tree.

## Dependencies

Core dependencies are the ones needed to define a net, train and test:
`numpy`, `torch`, `torch-geometric`, `gymnasium`, `simpn` and `dill`.
Everything else is an extra (`tensorboard`, `wandb`, `viz`), and the library
imports those lazily, inside the function that needs them, with an error or
warning that names the extra to install. Keep it that way: a new module-level
import of an optional package fails `test_imports_are_declared`.

`simpn` is pinned below 1.7. From 1.7 on, simpn depends on PyQt6 and imports
it from its simulator module, so `import gympn` fails on a headless Linux
machine without the system OpenGL libraries, and gympn's `Visualisation`
extends the pygame-based API of the 1.3-1.6 releases. From 1.8 on, a problem
cannot be deep-copied either, which gympn relies on.

## Documentation

The documentation lives in `docs/` and is built with MkDocs and the Material
theme. The API reference pages under `docs/reference/` are generated from the
docstrings by mkdocstrings, so documenting a public function means writing its
docstring. Preview locally with:

```bash
mkdocs serve
```

The site is deployed to <https://bpogroup.github.io/gympn> by
`.github/workflows/docs.yml` on every push to `main`.

## Pull requests

1. Fork the repository and create a branch.
2. Make the change, with a test when it changes behaviour.
3. Run the three checks above.
4. Open a pull request against `main` and describe what changes and why.

## Releasing

Releases are published to PyPI by `.github/workflows/release.yml` through
trusted publishing, so no API token lives in the repository. One-time setup
on GitHub and PyPI:

- On <https://test.pypi.org> and <https://pypi.org>, add a *trusted publisher*
  for the project `gympn` with owner `bpogroup`, repository `gympn`, workflow
  `release.yml` and environment `testpypi` or `pypi` respectively.
- In the GitHub repository settings, create the environments `testpypi` and
  `pypi`; add a required reviewer to `pypi` so publishing waits for approval.

To release version X.Y.Z:

1. Set `__version__ = "X.Y.Z"` in `gympn/__init__.py` and move the
   `Unreleased` entries of `CHANGELOG.md` under a new `X.Y.Z` heading.
2. Commit, then tag and push:

    ```bash
    git tag vX.Y.Z
    git push origin main vX.Y.Z
    ```

3. The workflow checks that the tag matches `__version__`, builds the sdist
   and the wheel, runs `twine check`, uploads to TestPyPI, and then waits for
   approval of the `pypi` environment before uploading to PyPI.

To check a build locally before tagging:

```bash
python -m build
twine check dist/*
```
