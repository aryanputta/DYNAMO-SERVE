.PHONY: install test lint bench dashboard clean

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --cov=. --cov-report=term-missing

lint:
	ruff check .
	mypy control_plane/ data_plane/ simulator/ ml/ --ignore-missing-imports

bench:
	python benchmarks/run_benchmarks.py --all --output results/

bench-quick:
	python benchmarks/run_benchmarks.py --scenario burst_traffic --output results/

dashboard:
	python dashboard/app.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; \
	find . -name "*.pyc" -delete; \
	rm -rf results/ .pytest_cache/ .mypy_cache/ htmlcov/ .coverage

demo:
	python benchmarks/run_benchmarks.py --scenario all --output results/ && \
	python dashboard/app.py
