.PHONY: test test-unit test-e2e coverage clean

test:
	pytest tests/ -v --no-cov

test-unit:
	PYTHONPATH=. pytest tests/unit/ -v --no-cov

test-e2e:
	pytest tests/e2e/ -v --timeout=120 --no-cov

coverage:
	PYTHONPATH=. pytest tests/unit/ -v --cov=src --cov-report=term-missing --cov-report=html:htmlcov --cov-report=xml:coverage.xml --cov-fail-under=90 --override-ini="addopts="

clean:
	rm -rf htmlcov coverage.xml .pytest_cache
