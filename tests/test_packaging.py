"""What shipping this repository as a package promises, checked rather than claimed.

Two promises, and both are load-bearing for the two-repository split.

**The engine has no dependencies.** ``pyproject.toml`` declares ``dependencies = []``, and
the whole architecture rests on it: a service can install the rules, the observation and the
public-information filter without pulling in torch's 191 MB. A single ``import numpy`` added
to ``catan/`` would break that silently — the developer machine has numpy, so nothing would
fail here.

**The wheel carries everything a consumer needs.** ``packages`` and ``package-data`` are
written by hand, on purpose, so that adding a package is a decision. The cost of that is a
new directory being quietly left out, which is what the tests below catch.

The thread section at the end is here for the same reason: it is a property of a
*deployment*, not of the game.
"""

import ast
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"


def declared():
    """``pyproject.toml``, parsed. ``tomllib`` is standard library from 3.11."""
    import tomllib

    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def imported_roots(path):
    """The top-level module names a file imports, from its syntax tree.

    Parsed rather than imported: importing to find out what something imports is circular,
    and it would also run module-level code.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


# =========================================================================== #
# THE ENGINE HAS NO DEPENDENCIES                                              #
# =========================================================================== #

def test_the_engine_imports_nothing_outside_the_standard_library():
    """⚠️ The single most important property of ``catan/``.

    The whole two-repository split rests on it: the website's API process installs the rules,
    the observation and the hidden-information filter and does **not** install torch, which
    costs 191 MB of RSS and 3.4 seconds of import time per process.

    A stray ``import numpy`` here would never fail on a developer machine, would never fail
    in the training suite, and would fail on the first deployment that took the promise
    literally.
    """
    allowed = set(sys.stdlib_module_names) | {"catan"}
    offenders = {}
    for path in sorted((ROOT / "catan").glob("*.py")):
        outside = imported_roots(path) - allowed
        if outside:
            offenders[path.name] = sorted(outside)
    assert not offenders, f"catan/ imports third-party modules: {offenders}"


def test_the_package_declares_no_dependencies():
    assert declared()["project"]["dependencies"] == []


def test_torch_is_an_extra_and_not_a_dependency():
    """``inference`` is the extra that adds it, so ``pip install catania-runtime`` gives a
    consumer the engine and nothing that weighs anything."""
    extras = declared()["project"]["optional-dependencies"]
    assert any("torch" in requirement for requirement in extras["inference"])
    assert not any("torch" in requirement for requirement in extras["web"])


def test_only_training_imports_torch():
    """The README's claim, held to. ``interfaces/`` reaches torch through ``training`` and
    always inside a ``try``, so a checkout without it still plays."""
    offenders = {}
    for path in sorted((ROOT / "catan").rglob("*.py")):
        if "torch" in imported_roots(path):
            offenders[str(path.relative_to(ROOT))] = "torch"
    assert not offenders, f"the engine imports torch: {offenders}"


# =========================================================================== #
# THE WHEEL CARRIES WHAT A CONSUMER NEEDS                                     #
# =========================================================================== #

def test_every_declared_package_exists():
    for name in declared()["tool"]["setuptools"]["packages"]:
        path = ROOT / pathlib.Path(*name.split("."))
        assert (path / "__init__.py").is_file(), f"{name} is declared and is not a package"


def test_no_importable_package_was_left_out_by_accident():
    """``packages`` is hand-written so that adding one is a decision. The cost of that is a
    new directory being silently omitted from the wheel, which is this test.

    ``benchmark`` and ``tests`` are excluded deliberately: they are development tools that
    work from a source checkout and have no business in a consumer's site-packages.
    """
    excluded = {"benchmark", "tests"}
    found = set()
    for init in ROOT.rglob("__init__.py"):
        relative = init.relative_to(ROOT).parent
        if relative.parts and relative.parts[0].startswith("."):
            continue
        found.add(".".join(relative.parts))
    missing = found - set(declared()["tool"]["setuptools"]["packages"]) - excluded
    assert not missing, f"packages exist and are not in pyproject.toml: {sorted(missing)}"


@pytest.mark.parametrize("package, pattern", [
    ("interfaces", "static/images/**/*.png"),
    ("interfaces.web", "static/*.html"),
    ("interfaces.web", "static/*.css"),
    ("interfaces.web", "static/*.js"),
])
def test_every_package_data_pattern_matches_something(package, pattern):
    """A glob that matches nothing produces a wheel with no board art, and the failure shows
    up as a website full of broken images rather than as an error."""
    root = ROOT / pathlib.Path(*package.split("."))
    assert list(root.glob(pattern)), f"{package}: {pattern} matched no files"


def test_the_declared_package_data_covers_the_art_the_client_asks_for():
    """The website serves the board art out of the installed package rather than keeping a
    second copy — the art set calls wheat `weat` and ore `stone`, and a second copy of that
    mapping is a bug waiting to happen."""
    patterns = declared()["tool"]["setuptools"]["package-data"]["interfaces"]
    shipped = {p for pattern in patterns for p in (ROOT / "interfaces").glob(pattern)}
    on_disk = set((ROOT / "interfaces" / "static" / "images").rglob("*.png"))
    assert on_disk, "there is no board art in this checkout"
    assert on_disk <= shipped, f"art not shipped: {sorted(on_disk - shipped)}"


def test_the_version_agrees_with_the_engine():
    """One version, two files, and they must not drift: ``catan.__version__`` is what
    ``contract.signature()`` publishes, and the wheel's version is what a consumer pins."""
    import catan

    assert declared()["project"]["version"] == catan.__version__


def test_the_python_floor_is_declared():
    """⚠️ Declared as 3.11 and *tested* only on the interpreter running this suite. The floor
    is a claim about syntax — no ``match``, no ``X | Y`` annotations anywhere in the tree —
    rather than a measurement, and it is written down here so the difference is visible.
    ``tomllib``, which this file uses, is itself 3.11+."""
    assert declared()["project"]["requires-python"] == ">=3.11"
    assert sys.version_info >= (3, 11)


# =========================================================================== #
# THE THREAD PIN                                                              #
# =========================================================================== #

def test_pinning_sets_every_variable_that_decides_the_pool_size():
    """Setting ``OMP_NUM_THREADS`` and not the others is a common way to half-fix this: which
    one matters depends on which BLAS torch was built against."""
    import os

    from training.threads import ENV_VARS, pin

    previous = {name: os.environ.get(name) for name in ENV_VARS}
    try:
        for name in ENV_VARS:
            os.environ.pop(name, None)
        pin(1)
        assert all(os.environ[name] == "1" for name in ENV_VARS)
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def test_pinning_reports_that_it_was_too_late_once_torch_is_imported():
    """⚠️ The failure this exists to make visible. Torch sizes its OpenMP pool when it is
    *imported*, so the environment variables set afterwards change nothing — and nothing
    raises. The measured cost of missing it is a median 1,758 ms against 179 ms.

    This suite has already imported torch, so the answer here is ``False`` and that is the
    point: a process that means to be single-threaded asserts on it at startup.
    """
    pytest.importorskip("torch")
    from training.threads import pin

    assert pin(1) is False


def test_pinning_does_not_overrule_an_operator_who_set_it_deliberately():
    import os

    from training.threads import pin

    previous = os.environ.get("OMP_NUM_THREADS")
    try:
        os.environ["OMP_NUM_THREADS"] = "4"
        pin(1)
        assert os.environ["OMP_NUM_THREADS"] == "4"
        pin(1, force=True)
        assert os.environ["OMP_NUM_THREADS"] == "1"
    finally:
        if previous is None:
            os.environ.pop("OMP_NUM_THREADS", None)
        else:
            os.environ["OMP_NUM_THREADS"] = previous


def test_pinning_children_narrows_the_environment_and_puts_it_back():
    """Children inherit ``os.environ`` at spawn, so the variables are set in the parent just
    before the pool is created and restored after — which pins the workers without narrowing
    the parent, whose own pool was sized at import."""
    import os

    from training.threads import ENV_VARS, for_children

    previous = os.environ.get(ENV_VARS[0])
    os.environ[ENV_VARS[0]] = "7"
    try:
        with for_children(1):
            assert all(os.environ[name] == "1" for name in ENV_VARS)
        assert os.environ[ENV_VARS[0]] == "7"
    finally:
        if previous is None:
            os.environ.pop(ENV_VARS[0], None)
        else:
            os.environ[ENV_VARS[0]] = previous
