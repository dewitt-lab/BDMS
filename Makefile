default:

install:
	pip install -e .[dev]

install-pyqt:
	pip install -e .[dev,pyqt]

test:
	pytest

doctest:
	pytest --doctest-modules

notebooks:
	pytest --nbval notebooks

format:
	black .
	docformatter --black --in-place **/*.py

lint:
	flake8 .

docs:
	make -C docs html

.PHONY: install test notebooks format lint docs
