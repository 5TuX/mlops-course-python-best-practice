# MLOps - Python Practices Course

## 1. Course Description 📚

This repository contains exercises related to the course on Python best practices, tools and workflows for MLOps.

The main goal here is to start from `src/main.py` and refactor it to a more maintainable and scalable codebase.

Some details about the codebase: a resnet model is loaded and used to make predictions on a sample image.
The label list is available here: https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a.

## 2. Installation

First, you can fork the repository to your own GitHub account. Then, you can clone the repository to your local machine:

```bash
git clone https://github.com/<your-github>/mlops-course-python-best-practice.git
```

Installing the dependencies depends on the choices you make. You can either use `venv`, `conda`, `poetry` or `uv`.

## 3. Usage

If you wish to start the application, without any changes, you can run the following command:

```bash
python src/main.py
```

Then it's up to you to refactor the codebase and make it more maintainable and scalable. You can apply the following tools :

- `ruff` or `black` for code formatting
- `mypy` for static type checking
- `.pre-commit` for pre-commit hooks
- `ruff` for linting
- Add some tests with `pytest`
- Compute the coverage with `pytest-cov`
- Add some documentation
