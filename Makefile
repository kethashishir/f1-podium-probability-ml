.PHONY: setup format lint test

setup:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

format:
	black .
	ruff check . --fix

lint:
	ruff check .

test:
	pytest -q
