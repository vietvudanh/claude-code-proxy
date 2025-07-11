.PHONY: fmt test lint

fmt:
	uvx ruff format

lint:
	uvx ruff check --select I --fix .
