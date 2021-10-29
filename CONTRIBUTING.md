# Contributing to this repository

## Getting started
- Before contributing, make sure you have a working developping environment set up.
```bash
    pip install tox
```
Few *tox* environments are defined for easier linting, testing and documentation generation.

We enforce strict coding rules. :
 - To make sure you comply with coding rules use the following command:
```bash
    tox -e isort
    tox -e flake8
```
 - Pyroma checks if the installation information is sufficient 
 ```bash
    tox -e pyroma
```

**DOES NOT WORK AT THE MOMENT:**
Automatic documentation can be generated like so:
```
    tox -e docs
```

For the entire workflow of linting, testing and documentation
```
    tox
```