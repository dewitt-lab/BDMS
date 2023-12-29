default:

install:
	pip install -e .[dev]

install-pyqt:
	pip install -e .[dev,pyqt]

lint:
	flake8 .
	black --check .
	docformatter --black **/*.py

format:
	black .
	docformatter --black --in-place **/*.py

test:
	pytest
	pytest --doctest-modules
	pytest --nbval docs/notebooks

docs:
	make -C docs html

.PHONY: install lint format test docs
