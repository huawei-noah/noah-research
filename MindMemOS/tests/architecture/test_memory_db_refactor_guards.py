import ast
from pathlib import Path

SRC_ROOT = Path("src/mindmemos/mindmemos")
INFRA_DB_ROOT = SRC_ROOT / "infra/db"
PIPELINES_ROOT = SRC_ROOT / "pipelines"

DB_BOUNDARY_ALLOW_PREFIXES = {
    SRC_ROOT / "pipelines/memory_db",
    SRC_ROOT / "pipelines/skill",
}
DB_BOUNDARY_ALLOW_FILES = {
    SRC_ROOT / "api/app.py",
    SRC_ROOT / "api/internal_routes.py",
    SRC_ROOT / "pipelines/dreaming/default.py",
    SRC_ROOT / "pipelines/feedback/implicit.py",
    SRC_ROOT / "workers/runtime.py",
}

RAW_CLIENT_ATTRS = {"qdrant", "neo4j"}
RAW_CLIENT_ALLOW_PREFIXES = DB_BOUNDARY_ALLOW_PREFIXES | {
    INFRA_DB_ROOT,
}
RAW_CLIENT_ALLOW_FILES = DB_BOUNDARY_ALLOW_FILES

CYPHER_ALLOW_PREFIXES = {
    INFRA_DB_ROOT,
    SRC_ROOT / "pipelines/memory_db",
    SRC_ROOT / "prompts",
}


def test_infra_db_does_not_import_business_layers() -> None:
    violations: list[str] = []
    forbidden_roots = ("mindmemos.api", "mindmemos.components", "mindmemos.pipelines", "mindmemos.llm")
    for path in INFRA_DB_ROOT.rglob("*.py"):
        for module in _imports(path):
            if module in forbidden_roots or module.startswith(tuple(f"{root}." for root in forbidden_roots)):
                violations.append(f"{path}: {module}")
    assert violations == []


def test_new_pipeline_code_does_not_reach_low_level_database_clients_directly() -> None:
    violations: list[str] = []
    for path in PIPELINES_ROOT.rglob("*.py"):
        if _allowed(path, DB_BOUNDARY_ALLOW_PREFIXES, DB_BOUNDARY_ALLOW_FILES):
            continue
        text = path.read_text(encoding="utf-8")
        if "get_database_clients" in text or "limit_database_clients" in text:
            violations.append(str(path))
    assert violations == []


def test_raw_qdrant_and_neo4j_client_attributes_stay_inside_database_boundary() -> None:
    violations: list[str] = []
    for path in SRC_ROOT.rglob("*.py"):
        if _allowed(path, RAW_CLIENT_ALLOW_PREFIXES, RAW_CLIENT_ALLOW_FILES):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in RAW_CLIENT_ATTRS:
                violations.append(f"{path}: .{node.attr}")
    assert violations == []


def test_raw_cypher_literals_stay_inside_graph_boundaries() -> None:
    violations: list[str] = []
    cypher_markers = ("MATCH ", "MERGE ", "UNWIND ", "RETURN ")
    for path in SRC_ROOT.rglob("*.py"):
        if _allowed(path, CYPHER_ALLOW_PREFIXES, set()):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if any(marker in node.value for marker in cypher_markers):
                    violations.append(str(path))
                    break
    assert violations == []


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(_absolute_import(path, node.module, node.level))
    return modules


def _absolute_import(path: Path, module: str, level: int) -> str:
    if level <= 0:
        return module
    package_parts = path.relative_to(SRC_ROOT).with_suffix("").parts[:-1]
    base_parts = ("mindmemos", *package_parts[: max(0, len(package_parts) - level + 1)])
    return ".".join((*base_parts, module))


def _allowed(path: Path, prefixes: set[Path], files: set[Path]) -> bool:
    return path in files or any(_is_relative_to(path, prefix) for prefix in prefixes)


def _is_relative_to(path: Path, prefix: Path) -> bool:
    try:
        path.relative_to(prefix)
        return True
    except ValueError:
        return False
