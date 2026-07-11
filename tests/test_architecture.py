"""The dependency DAG (docs/ARCHITECTURE.md "Dependency rules (acyclic)") is enforced,
not aspirational: lint-imports validates every [tool.importlinter] contract in
pyproject.toml against the real import graph."""
from __future__ import annotations

import subprocess


def test_dependency_dag_contracts():
    """Every import-linter contract holds; on failure, print the broken
    contract and the offending import chain verbatim."""
    res = subprocess.run(["lint-imports"], capture_output=True, text=True)
    assert res.returncode == 0, (
        f"dependency DAG violated:\n{res.stdout}\n{res.stderr}"
    )
