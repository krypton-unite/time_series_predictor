# Time Series Predictor

[![Documentation Status](https://readthedocs.org/projects/timeseriespredictor/badge/?version=latest)](https://timeseriespredictor.readthedocs.io/en/latest/?badge=latest) [![travis](https://travis-ci.org/DanielAtKrypton/time_series_predictor.svg?branch=master)](https://travis-ci.org/github/DanielAtKrypton/time_series_predictor) [![codecov](https://codecov.io/gh/DanielAtKrypton/time_series_predictor/branch/master/graph/badge.svg)](https://codecov.io/gh/DanielAtKrypton/time_series_predictor)

## Development

### Pre requisistes installation

#### Windows

```powershell
virtualenv -p python3 .env
.\.env\Scripts\activate
pip install -e .[dev]
```

#### Linux or MacOS

```powershell
virtualenv -p python3 .env
. .env/bin/activate
pip install -e .[dev]
```

------

### Test

```terminal
pytest -s
```

------

### Build docs

```terminal
cd docs
make html
sphinx-build -b rinoh source build/rinoh
```
