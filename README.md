# Base-Python

## Project Setup

This project uses [uv](https://github.com/astral-sh/uv) for Python package management. Follow these steps to set up your development environment:

Run the following commands:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone repository and navigate to project directory
git clone <repository-url>
cd my-project

# Create and activate virtual environment
uv venv
source .venv/bin/activate

# Install project dependencies
uv pip install -e ".[dev]"

# Install and enable JupyterLab code formatter
jupyter labextension install @ryantam626/jupyterlab_code_formatter
jupyter serverextension enable --py jupyterlab_code_formatter
```

This will install all required packages listed in `pyproject.toml`, including both main dependencies and development tools.

## Pre-commit Hooks Setup

This project uses [pre-commit](https://pre-commit.com/) to maintain code
quality. Pre-commit runs a series of checks and formatters on your code before
each commit, ensuring consistent style and catching common issues early.

### Required Libraries and Tools

You need to install the following dependencies:

```bash
pip install pre-commit black isort ruff mypy
npm install -g prettier
brew install hadolint
pre-commit install
```
