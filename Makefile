.PHONY: test bench demo lint matrix all

all: lint test

test:            ## run the self-test suite
	python -m format.selftest

bench:           ## full benchmarks; writes results/bench-<timestamp>.{md,json}
	python -m format.bench

bench-quick:     ## fewer samples, still writes a report
	python -m format.bench --runs 5 --nodes 20

sweeps:          ## just the nu / d / p parameter sweeps
	python -m format.bench --section sweeps

demo:            ## unicast, churn, reply, multicast
	python demo.py

lint:
	ruff check .

matrix:          ## self-tests across several parameter sets
	pytest
