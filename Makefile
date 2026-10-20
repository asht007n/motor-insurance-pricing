# Shortcuts for macOS/Linux. On Windows, run the commands in the README directly.

.PHONY: install data model test all clean

install:
	pip install -e ".[dev]"

data:
	python -m mtpl.data

model:
	python -m mtpl.pipeline

test:
	pytest -q

all: data model test

clean:
	rm -rf data/
