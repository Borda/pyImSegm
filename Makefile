.PHONY: test clean docs

clean:
	# clean all temp runs
	rm -rf .pytest_cache
	rm -rf ./docs/build
	rm -rf ./docs/source/*.md
	rm -rf ./docs/source/api
	rm -rf ./docs/source/notebooks

test: clean
	pip install --quiet -r requirements.txt
	pip install --quiet -r tests/requirements.txt

	# run tests with coverage
	python -m coverage run --source imsegm -m pytest imsegm tests -v
	python -m coverage report

docs: clean
	pip install --quiet -r docs/requirements.txt
	python setup.py build_ext --inplace
	python -m sphinx -b html -W --keep-going docs/source docs/build
