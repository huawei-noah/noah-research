import ast
from pathlib import Path

ROOT = Path("src/mindmemos/mindmemos")
CHECK_ROOTS = [ROOT / "api", ROOT / "components", ROOT / "pipelines"]
ALLOWLIST = {
    ROOT / "api/app.py",
}
ALLOW_PREFIXES = {
    ROOT / "pipelines/memory_db",
    ROOT / "pipelines/skill",
}
FORBIDDEN_SNIPPETS = [
    "get_database_clients",
    "from mindmemos.infra.db import SparseVectorData",
    "from mindmemos.infra.db.registry import get_database_clients",
]


def test_business_layers_do_not_import_low_level_db_clients() -> None:
    violations: list[str] = []
    for root in CHECK_ROOTS:
        for path in root.rglob("*.py"):
            normalized = Path(path.as_posix())
            if normalized in ALLOWLIST or any(_is_relative_to(normalized, prefix) for prefix in ALLOW_PREFIXES):
                continue
            text = path.read_text(encoding="utf-8")
            for snippet in FORBIDDEN_SNIPPETS:
                if snippet in text:
                    violations.append(f"{path}: {snippet}")
    assert violations == []


def test_core_package_does_not_import_application_adapters() -> None:
    violations: list[str] = []
    for path in ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            imported_modules: list[str] = []
            if isinstance(node, ast.Import):
                imported_modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.append(node.module)
            for module in imported_modules:
                if module == "application" or module.startswith("application."):
                    violations.append(f"{path}: {module}")
    assert violations == []


def _is_relative_to(path: Path, prefix: Path) -> bool:
    try:
        path.relative_to(prefix)
        return True
    except ValueError:
        return False
