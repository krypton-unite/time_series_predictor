# Time Series Predictor

[![PyPI version](https://badge.fury.io/py/time-series-predictor.svg)](https://badge.fury.io/py/time-series-predictor) [![Documentation Status](https://readthedocs.org/projects/time-series-predictor/badge/?version=latest)](https://time-series-predictor.readthedocs.io/en/latest/?badge=latest) [![travis](https://travis-ci.org/krypton-unite/time_series_predictor.svg?branch=master)](https://travis-ci.org/github/krypton-unite/time_series_predictor) [![codecov](https://codecov.io/gh/krypton-unite/time_series_predictor/branch/master/graph/badge.svg)](https://codecov.io/gh/krypton-unite/time_series_predictor) [![Requirements Status](https://requires.io/github/krypton-unite/time_series_predictor/requirements.svg?branch=master)](https://requires.io/github/krypton-unite/time_series_predictor/requirements/?branch=master) [![GitHub license](https://img.shields.io/github/license/krypton-unite/time_series_predictor)](https://github.com/krypton-unite/time_series_predictor)

## Usage

Please refer to the following examples:

- [example_flights_dataset.ipynb](https://time-series-predictor.readthedocs.io/en/latest/notebooks/example_flights_dataset.html)
- [example_oze_challenge.ipynb](https://time-series-predictor.readthedocs.io/en/latest/notebooks/example_oze_challenge.html)

## Pre requisistes installation

### Windows

```powershell
python -m venv .\.env
.\.env\Scripts\activate
pip install pip-tools
python setup.py synchronize
```

#### Linux or MacOS

```bash
python -m venv .env
. .env/bin/activate
pip install pip-tools
python setup.py synchronize
```

--------

## Development

```terminal
pip install -e .[dev]
```

--------

## Test

```terminal
pip install -e .[test]
pytest -s
```

--------

## Build docs

```terminal
pip install -e .[docs]
# cd docs
# make html
sphinx-build docs/source docs/build
```
