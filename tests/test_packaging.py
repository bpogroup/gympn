"""Packaging and dependency checks.

These tests check that gympn installs and runs with exactly the dependencies
that pyproject.toml declares:

- every third-party import in the library is declared, either as a core
  dependency or, for imports done lazily inside functions, as an extra;
- ``import gympn`` does not load any optional dependency;
- training and testing work with the optional dependencies missing;
- the package data (the logo, py.typed) is installed;
- the wheel contains the library only, and its metadata matches pyproject;
- the examples import only names that exist.

The wheel-building tests are marked ``slow``; deselect them with
``pytest -m "not slow"``. The fresh-environment install test downloads torch,
so it runs only when GYMPN_INSTALL_TEST=1.
"""

import ast
import importlib
import importlib.metadata
import importlib.resources
import os
import re
import subprocess
import sys
import textwrap
import venv
import zipfile
from pathlib import Path

import pytest

import gympn

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    tomllib = pytest.importorskip("tomli")

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "gympn"
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf8"))

# Distribution name -> top-level import name, where they differ.
IMPORT_NAME = {"torch-geometric": "torch_geometric"}

# Third-party modules that come with a declared dependency instead of being
# declared themselves (simpn depends on pygame; its visualisation uses it).
PROVIDED_BY = {"pygame": "simpn"}

# Extras that must stay optional: `import gympn` must not load them.
OPTIONAL_MODULES = ["wandb", "tensorboard", "networkx"]

# Library modules that exist only to serve one extra. They may import that
# extra at module level, but the rest of gympn may import them only lazily.
EXTRA_MODULES = {"plotter": "viz", "wandb_integration": "wandb"}


def _dist_name(requirement):
    return re.split(r"[\s<>=!~;\[]", requirement, maxsplit=1)[0].lower()


def _import_name(requirement):
    name = _dist_name(requirement)
    return IMPORT_NAME.get(name, name.replace("-", "_"))


CORE = {_import_name(r) for r in PYPROJECT["project"]["dependencies"]}
EXTRAS = {
    extra: {_import_name(r) for r in reqs if not r.startswith("gympn")}
    for extra, reqs in PYPROJECT["project"]["optional-dependencies"].items()
}
ALL_DECLARED = CORE.union(*EXTRAS.values())


def _third_party_imports(path):
    """(module-level, function-level) sets of third-party top-level imports."""
    tree = ast.parse(path.read_text(encoding="utf8"))
    top, lazy = set(), set()

    def visit(node, inside_function):
        for child in ast.iter_child_nodes(node):
            nested = inside_function or isinstance(
                child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
            if isinstance(child, ast.Import):
                names = [a.name for a in child.names]
            elif isinstance(child, ast.ImportFrom) and child.level == 0:
                names = [child.module]
            else:
                names = []
            for name in names:
                root = name.split(".")[0]
                if root in sys.stdlib_module_names or root == "gympn":
                    continue
                (lazy if nested else top).add(root)
            visit(child, nested)

    visit(tree, False)
    return top, lazy


def _run_python(code, cwd, env=None, timeout=600):
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code)], cwd=cwd,
                          capture_output=True, text=True, timeout=timeout,
                          env={**os.environ, **(env or {})})


# ---------------------------------------------------------------------------
# Declared dependencies


def test_version_matches_installed_metadata():
    try:
        installed = importlib.metadata.version("gympn")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("gympn is not installed (run `pip install -e .`)")
    assert installed == gympn.__version__


def test_installed_requirements_match_pyproject():
    try:
        requires = importlib.metadata.requires("gympn") or []
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("gympn is not installed (run `pip install -e .`)")
    core = {_dist_name(r) for r in requires if "extra ==" not in r}
    assert core == {_dist_name(r) for r in PYPROJECT["project"]["dependencies"]}, (
        "the installed metadata is stale; reinstall with `pip install -e .`")


@pytest.mark.parametrize("module", sorted(CORE))
def test_core_dependency_importable(module):
    importlib.import_module(module)


def test_test_tools_are_not_runtime_dependencies():
    core = {_dist_name(r) for r in PYPROJECT["project"]["dependencies"]}
    assert not core & {"pytest", "ruff", "build", "twine", "mkdocs-material"}


@pytest.mark.parametrize("path", sorted(PACKAGE.glob("*.py")), ids=lambda p: p.name)
def test_imports_are_declared(path):
    """Module-level imports need a core dependency; lazy ones at least an extra."""
    top, lazy = _third_party_imports(path)
    provided = {m for m, dist in PROVIDED_BY.items() if _import_name(dist) in CORE}
    allowed_top = CORE | provided | EXTRAS.get(EXTRA_MODULES.get(path.stem), set())
    undeclared_top = top - allowed_top
    assert not undeclared_top, (
        f"{path.name} imports {sorted(undeclared_top)} at module level, but they "
        f"are not core dependencies; declare them or import them lazily")
    undeclared = lazy - ALL_DECLARED - provided
    assert not undeclared, f"{path.name} imports undeclared {sorted(undeclared)}"


def _module_level_gympn_imports(path):
    tree = ast.parse(path.read_text(encoding="utf8"))
    found = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            if node.level == 1 and node.module:
                found.add(node.module.split(".")[0])
            elif node.module and node.module.startswith("gympn."):
                found.add(node.module.split(".")[1])
        elif isinstance(node, ast.Import):
            found |= {a.name.split(".")[1] for a in node.names if a.name.startswith("gympn.")}
    return found


@pytest.mark.parametrize("path", sorted(PACKAGE.glob("*.py")), ids=lambda p: p.name)
def test_extra_modules_are_imported_lazily(path):
    eager = _module_level_gympn_imports(path) & set(EXTRA_MODULES)
    assert not eager, f"{path.name} imports {sorted(eager)} at module level"


def test_every_declared_dependency_is_used():
    used = set()
    for path in PACKAGE.glob("*.py"):
        top, lazy = _third_party_imports(path)
        used |= top | lazy
    # torch.utils.tensorboard is imported through torch, so the tensorboard
    # extra counts as used when that import is present.
    if "from torch.utils.tensorboard" in (PACKAGE / "agents.py").read_text(encoding="utf8"):
        used.add("tensorboard")
    runtime = CORE | EXTRAS["tensorboard"] | EXTRAS["wandb"] | EXTRAS["viz"]
    assert not runtime - used, f"declared but never imported: {sorted(runtime - used)}"


# ---------------------------------------------------------------------------
# Optional dependencies stay optional


def test_import_does_not_load_optional_dependencies():
    result = _run_python(f"""
        import sys, gympn
        loaded = [m for m in {OPTIONAL_MODULES!r} if m in sys.modules]
        assert not loaded, loaded
        """, cwd=ROOT)
    assert result.returncode == 0, result.stderr


# Makes the optional modules look absent, as if their extras were not
# installed: the path finder reports them as not found, which is what both
# `import` and importlib.util.find_spec (used by torch to probe for them) see.
_BLOCK_OPTIONAL = """
    import sys
    from importlib.machinery import PathFinder
    _find_spec = PathFinder.find_spec.__func__
    def _blocked_find_spec(cls, name, path=None, target=None):
        if name.split('.')[0] in {blocked!r} or name == 'torch.utils.tensorboard':
            return None
        return _find_spec(cls, name, path, target)
    PathFinder.find_spec = classmethod(_blocked_find_spec)
"""

# A two-employee task assignment problem, small enough to train in seconds.
_TINY_PROBLEM = """
    from simpn.simulator import SimToken
    from gympn import GymProblem, RandomSolver

    def make_problem():
        p = GymProblem()
        arrival = p.add_var("arrival", var_attributes=["task_type"])
        waiting = p.add_var("waiting", var_attributes=["task_type"])
        busy = p.add_var("busy", var_attributes=["task_type", "resource_id"])
        employee = p.add_var("employee", var_attributes=["code_employee"])
        arrival.put({"task_type": 0})
        arrival.put({"task_type": 1})
        employee.put({"code_employee": 0})
        employee.put({"code_employee": 1})
        p.add_event([arrival], [arrival, waiting], name="arrive",
                    behavior=lambda a: [SimToken(a, delay=1), SimToken(a)])
        p.add_action([waiting, employee], [busy], name="start",
                     behavior=lambda c, r: [SimToken((c, r), delay=1 if
                                            c["task_type"] == r["code_employee"] else 2)])
        p.add_event([busy], [employee], lambda b: [SimToken(b[1])], name="complete",
                    reward_function=lambda x: 1)
        return p
"""


def test_train_and_test_without_optional_dependencies(tmp_path):
    """A full training run and a test run with wandb, tensorboard and networkx missing."""
    code = _BLOCK_OPTIONAL.format(blocked=OPTIONAL_MODULES) + _TINY_PROBLEM + f"""
    import warnings
    warnings.simplefilter("always")
    problem = make_problem()
    problem.training_run(length=10, args_dict={{
        "epochs": 1, "episodes": 2, "policy_updates": 1, "value_updates": 1,
        "logdir": {str(tmp_path)!r}, "name": "run", "datetag": False,
        "use_wandb": True, "open_tensorboard": False, "verbose": 0,
    }})
    for m in {OPTIONAL_MODULES!r}:
        assert m not in sys.modules, m
    reward = make_problem().testing_run(RandomSolver(), length=10)
    print("REWARD", reward)
    """
    result = _run_python(code, cwd=tmp_path)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    assert "REWARD" in result.stdout
    output = result.stdout + result.stderr
    assert "gympn[tensorboard]" in output, "missing tensorboard should be reported"
    assert list(tmp_path.rglob("*.pth")), "training saved no policy"


def test_plotting_without_viz_extra_fails_with_clear_error():
    code = _BLOCK_OPTIONAL.format(blocked=OPTIONAL_MODULES) + """
    from gympn import GymProblem
    try:
        GymProblem(plot_observations=True)
    except ImportError as e:
        print("IMPORTERROR", e)
    """
    result = _run_python(code, cwd=ROOT)
    assert "IMPORTERROR" in result.stdout, result.stdout + result.stderr
    assert "gympn[viz]" in result.stdout


# ---------------------------------------------------------------------------
# Package data


def test_package_data_installed():
    files = importlib.resources.files("gympn")
    assert files.joinpath("assets", "logo.png").is_file()
    assert files.joinpath("py.typed").is_file()


# ---------------------------------------------------------------------------
# Distributions


@pytest.fixture(scope="module")
def wheel(tmp_path_factory):
    out = tmp_path_factory.mktemp("dist")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "wheel", str(ROOT), "--no-deps",
         "--no-build-isolation", "-w", str(out), "-q"],
        capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stderr
    (path,) = out.glob("gympn-*.whl")
    return path


@pytest.mark.slow
def test_wheel_contains_only_the_library(wheel):
    names = zipfile.ZipFile(wheel).namelist()
    stray = [n for n in names if not n.startswith(("gympn/", "gympn-"))]
    assert not stray, f"files outside the package: {stray[:10]}"
    assert "gympn/assets/logo.png" in names
    assert "gympn/py.typed" in names
    assert not [n for n in names if n.endswith((".pth", ".pyc", ".log"))]
    assert wheel.stat().st_size < 2_000_000


@pytest.mark.slow
def test_wheel_metadata(wheel):
    meta = zipfile.ZipFile(wheel).read(
        f"gympn-{gympn.__version__}.dist-info/METADATA").decode()
    assert f"Version: {gympn.__version__}" in meta
    assert "Requires-Python: >=3.10" in meta
    requires = [line.split(":", 1)[1].strip() for line in meta.splitlines()
                if line.startswith("Requires-Dist:")]
    core = {_dist_name(r) for r in requires if "extra ==" not in r}
    assert core == {_dist_name(r) for r in PYPROJECT["project"]["dependencies"]}
    for extra in ("tensorboard", "wandb", "viz", "all", "dev", "docs"):
        assert f"Provides-Extra: {extra}" in meta


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("GYMPN_INSTALL_TEST") != "1",
                    reason="downloads torch; set GYMPN_INSTALL_TEST=1 to run")
def test_fresh_environment_install(wheel, tmp_path):
    """Install the wheel without extras in a new venv, then train and test."""
    env_dir = tmp_path / "venv"
    venv.create(env_dir, with_pip=True)
    python = env_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    install = subprocess.run(
        [str(python), "-m", "pip", "install", "-q", str(wheel),
         "--extra-index-url", "https://download.pytorch.org/whl/cpu"],
        capture_output=True, text=True, timeout=1800)
    assert install.returncode == 0, install.stderr[-3000:]
    code = _TINY_PROBLEM + f"""
    import sys
    problem = make_problem()
    problem.training_run(length=10, args_dict={{
        "epochs": 1, "episodes": 2, "logdir": {str(tmp_path / "train")!r},
        "name": "run", "datetag": False, "open_tensorboard": False, "verbose": 0}})
    print("REWARD", make_problem().testing_run(RandomSolver(), length=10))
    for m in {OPTIONAL_MODULES!r}:
        assert m not in sys.modules, m
    """
    run = subprocess.run([str(python), "-c", textwrap.dedent(code)], cwd=tmp_path,
                         capture_output=True, text=True, timeout=900)
    assert run.returncode == 0, run.stdout[-3000:] + run.stderr[-3000:]
    assert "REWARD" in run.stdout


# ---------------------------------------------------------------------------
# Examples


@pytest.mark.parametrize("path", sorted((ROOT / "examples").glob("*.py")),
                         ids=lambda p: p.name)
def test_example_imports_resolve(path):
    """Every `from gympn... import name` in the examples names something that exists."""
    tree = ast.parse(path.read_text(encoding="utf8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("gympn"):
            module = importlib.import_module(node.module)
            for alias in node.names:
                assert hasattr(module, alias.name), f"{node.module}.{alias.name}"
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("gympn"):
                    importlib.import_module(alias.name)
