"""Test-suite conventions.

- Layout mirrors ``src/query_pipeline``: each package gets a ``tests/<pkg>/``
  directory with one ``test_<module>.py`` per module (or per coherent concern).
- ``src`` is importable via ``pythonpath = ["src"]`` (pyproject.toml); the
  package is also installed editable in the project venv.
- Duplicate basenames (e.g. ``test_assemble.py`` under ``prompts/`` and
  ``session/``) are handled by ``--import-mode=importlib`` (pyproject.toml).
- Tests that need repository-relative paths (``templates/``, ``configs/``)
  derive them as ``ROOT = Path(__file__).resolve().parents[2]`` at module
  level, since ``unittest.TestCase`` methods cannot take pytest fixtures.
"""
