# Time Series Predictor

[![Documentation Status](https://readthedocs.org/projects/timeseriespredictor/badge/?version=latest)](https://timeseriespredictor.readthedocs.io/en/latest/?badge=latest) [![travis](https://travis-ci.org/DanielAtKrypton/time_series_predictor.svg?branch=master)](https://travis-ci.org/github/DanielAtKrypton/time_series_predictor) [![codecov](https://codecov.io/gh/DanielAtKrypton/time_series_predictor/branch/master/graph/badge.svg)](https://codecov.io/gh/DanielAtKrypton/time_series_predictor) [![GitHub license](https://img.shields.io/github/license/DanielAtKrypton/time_series_predictor)](https://github.com/DanielAtKrypton/time_series_predictor)

## Usage

Please refer to the following [jupyter notebook](https://github.com/DanielAtKrypton/time_series_predictor/blob/master/docs/source/notebooks/example_flights_dataset.ipynb) as a usage example.

## Development

### Pre requisistes installation

#### Windows

```powershell
virtualenv -p python3 .env
.\.env\Scripts\activate
pip install -e .[dev]
```

#### Linux or MacOS

```bash
virtualenv -p python3 .env
. .env/bin/activate
pip install -e .[dev]
```

------

### Test

```bash
pip install -e .[test]
pytest -s
```

------

### Build docs

```bash
pip install -e .[docs]
cd docs
make html
```
