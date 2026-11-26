"""The deployed app must install from requirements.txt alone.

Streamlit Community Cloud installs requirements.txt and nothing else, while a
development machine usually has requirements-pipeline.txt installed too. So an
`import sklearn` added to app.py would work locally and break only once
deployed. These tests read the source instead of importing it: they follow
app.py through every `src` module it imports and check that each third-party
package it reaches is listed in requirements.txt.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Import name -> distribution name, where the two differ.
DISTRIBUTION = {
        "sklearn": "scikit-learn",
        "umap": "umap-learn",
        "yaml": "pyyaml",
        "PIL": "pillow",
        "bs4": "beautifulsoup4",
}


def requirement_names(path: Path) -> set[str]:
        """Distribution names listed in one requirements file, following -r."""
        names: set[str] = set()
        for line in path.read_text(encoding="utf-8").splitlines():
                line = line.split("#", 1)[0].strip()
                if not line:
                        continue
                if line.startswith("-r"):
                        names |= requirement_names(path.parent / line[2:].strip())
                        continue
                name = re.split(r"[<>=!~;\[ ]", line, maxsplit=1)[0]
                names.add(name.lower().replace("_", "-"))
        return names


def module_path(dotted: str) -> Path | None:
        """File for a `src...` module, or None if it is not a module file."""
        base = ROOT.joinpath(*dotted.split("."))
        for candidate in (base.with_suffix(".py"), base / "__init__.py"):
                if candidate.exists():
                        return candidate
        return None


def app_imports() -> tuple[set[str], list[str]]:
        """Top-level third-party packages reachable from app.py.

        Returns the packages and the files that were scanned. Imports inside
        functions count too, because they run as soon as the app calls them.
        """
        packages: set[str] = set()
        seen: set[Path] = set()
        queue: list[tuple[Path, str]] = [(ROOT / "app.py", "")]

        while queue:
                path, package = queue.pop()
                if path in seen:
                        continue
                seen.add(path)
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

                for node in ast.walk(tree):
                        targets: list[str] = []
                        if isinstance(node, ast.Import):
                                targets = [alias.name for alias in node.names]
                        elif isinstance(node, ast.ImportFrom):
                                if node.level:
                                        parent = package.rsplit(".", node.level - 1)[0]
                                        base = f"{parent}.{node.module}" if node.module else parent
                                else:
                                        base = node.module or ""
                                # `from src import radar` may name submodules.
                                targets = [base] + [f"{base}.{alias.name}" for alias in node.names]

                        for name in targets:
                                top = name.split(".")[0]
                                if top == "src":
                                        found = module_path(name)
                                        if found is not None:
                                                parent = name if found.name == "__init__.py" else name.rsplit(".", 1)[0]
                                                queue.append((found, parent))
                                elif top and top != "__future__" and top not in sys.stdlib_module_names:
                                        packages.add(top)

        return packages, sorted(str(p.relative_to(ROOT)) for p in seen)


def test_app_imports_only_what_requirements_txt_installs():
        listed = requirement_names(ROOT / "requirements.txt")
        packages, scanned = app_imports()
        missing = sorted(
                p for p in packages
                if DISTRIBUTION.get(p, p).lower().replace("_", "-") not in listed
        )
        assert not missing, (
                f"The app imports {missing}, which requirements.txt does not install, "
                f"so the deployed app would fail. Scanned: {scanned}. Either add them "
                f"to requirements.txt or keep them out of the app's import chain."
        )


def test_app_import_scan_reaches_the_src_modules():
        """Guards the guard: a scan that silently stops at app.py proves nothing."""
        packages, scanned = app_imports()
        assert "app.py" in scanned
        assert any(path.startswith("src") for path in scanned), scanned
        assert {"streamlit", "pandas", "plotly"} <= packages


def test_pipeline_requirements_include_the_app_requirements():
        text = (ROOT / "requirements-pipeline.txt").read_text(encoding="utf-8")
        assert re.search(r"^-r requirements\.txt\s*$", text, flags=re.MULTILINE)
        assert requirement_names(ROOT / "requirements.txt") <= requirement_names(
                ROOT / "requirements-pipeline.txt"
        )
