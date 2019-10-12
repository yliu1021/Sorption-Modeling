.PHONY: env
env:
	python3 -m venv env
	env/bin/pip install --upgrade pip
	env/bin/pip install --upgrade setuptools
	env/bin/pip install pylint
	env/bin/pip install -r requirements.txt
