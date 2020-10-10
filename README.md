# Time Series Predictor

[![PyPI version](https://badge.fury.io/py/time-series-predictor.svg)](https://badge.fury.io/py/time-series-predictor) [![Documentation Status](https://readthedocs.org/projects/timeseriespredictor/badge/?version=latest)](https://timeseriespredictor.readthedocs.io/en/latest/?badge=latest) [![travis](https://travis-ci.org/DanielAtKrypton/time_series_predictor.svg?branch=master)](https://travis-ci.org/github/DanielAtKrypton/time_series_predictor) [![codecov](https://codecov.io/gh/DanielAtKrypton/time_series_predictor/branch/master/graph/badge.svg)](https://codecov.io/gh/DanielAtKrypton/time_series_predictor) [![Requirements Status](https://requires.io/github/DanielAtKrypton/time_series_predictor/requirements.svg?branch=master)](https://requires.io/github/DanielAtKrypton/time_series_predictor/requirements/?branch=master) [![GitHub license](https://img.shields.io/github/license/DanielAtKrypton/time_series_predictor)](https://github.com/DanielAtKrypton/time_series_predictor)

## Usage

Please refer to the following examples:

- [example_flights_dataset.ipynb](https://github.com/DanielAtKrypton/time_series_predictor/blob/master/docs/source/notebooks/example_flights_dataset.ipynb)
- [example_oze_challenge.ipynb](https://github.com/DanielAtKrypton/time_series_predictor/blob/master/docs/source/notebooks/example_oze_challenge.ipynb)

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
# cd app/docs
# make html
sphinx-build app/docs/source app/docs/build
```
