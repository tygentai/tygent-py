# Agent Working Guidelines

This repository uses automated tests and code formatting. When modifying any code or documentation:

- Install the project in editable mode if dependencies are missing:
  ```bash
  pip install -e .
  ```
- Run the test suite:
  ```bash
  pytest -q
  ```
- Format Python code with [Black](https://black.readthedocs.io/) and ensure imports are ordered with [isort](https://pycqa.github.io/isort/):
  ```bash
  black .
  isort .
  ```
- When creating pull requests, include a short summary of key changes and test results.

These guidelines apply to all files in this repository.
