# Makefile with phony targets

.PHONY: setup lint format setup-hooks

setup: ## Setup Python Environment
	@make setup-git-hooks
	cp .env.example .env
	poetry install
	pip install -e .

setup-ci: ## Setup the environment for CI
	curl -sSL https://install.python-poetry.org | POETRY_VERSION=1.4.2 python -
	cp .env.example .env
	poetry install

test: ## test
	@poetry shell
	pytest

lint: ## Lint Python Files
	@poetry run ruff check .

format: ## Format Python Files
	@poetry run ruff check --fix .

setup-git-hooks: ## Setup Git Hook
	@echo "Setting up Git hooks..."
	cp .git-hooks/pre-commit .git/hooks/pre-commit
	chmod +x .git/hooks/pre-commit

help: ## show help message
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m\033[0m\n"} /^[$$()% a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)
