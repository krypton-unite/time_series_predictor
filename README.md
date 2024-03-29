# Time Series Predictor

[![PyPI version](https://badge.fury.io/py/time-series-predictor.svg)](https://badge.fury.io/py/time-series-predictor) [![Documentation Status](https://readthedocs.org/projects/time-series-predictor/badge/?version=latest)](https://time-series-predictor.readthedocs.io/en/latest/?badge=latest) [![travis](https://app.travis-ci.com/krypton-unite/time_series_predictor.svg?branch=master)](https://app.travis-ci.com/github/krypton-unite/time_series_predictor) [![codecov](https://codecov.io/gh/krypton-unite/time_series_predictor/branch/master/graph/badge.svg)](https://codecov.io/gh/krypton-unite/time_series_predictor) [![GitHub license](https://img.shields.io/github/license/krypton-unite/time_series_predictor)](https://github.com/krypton-unite/time_series_predictor) [![Requirements Status](https://requires.io/github/krypton-unite/time_series_predictor/requirements.svg?branch=master)](https://requires.io/github/krypton-unite/time_series_predictor/requirements/?branch=master)

## Usage

Please refer to the following examples:

- [example_flights_dataset.ipynb](https://time-series-predictor.readthedocs.io/en/latest/notebooks/example_flights_dataset.html)
- [example_oze_challenge.ipynb](https://time-series-predictor.readthedocs.io/en/latest/notebooks/example_oze_challenge.html)

## Pre requisistes installation

### Windows

```powershell
.\scrips\init-env.ps1
```

#### Linux or MacOS

```bash
./scripts/init-env.sh
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

### Windows Requirements

#### Make

In an elevated powershell prompt install it by:

```powershell
choco install cmake
```

#### TexLive

Download and install TexLive from [ctan.org](http://mirror.ctan.org/systems/texlive/tlnet/install-tl-windows.exe)

#### Perl requirement

In an elevated powershell prompt install it by:

```powershell
choco install activeperl
```

#### CPANM requirement

In an elevated powershell prompt install it by:

```powershell
curl -L https://cpanmin.us | perl - App::cpanminus
```

#### Latex indent requirement

```powershell
git clone https://github.com/cmhughes/latexindent.pl.git
cd latexindent.pl/helper-scripts
perl latexindent-module-installer.pl
```

#### ImageMagick requirement

In an elevated powershell prompt install it by:

```powershell
choco install imagemagick
```

----------------

### Build process

```powershell
pip install -e .[docs]
# cd docs
.\make html
.\make latex
# sphinx-build docs/source docs/build
.\build\latex\make
```
