"""
Setup
"""
import os
import setuptools


def package_files(directory):
    """package_files

    recursive method which will lets you set the
    package_data parameter in the setup call.
    """
    paths = []
    for (path, _, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths

extra_files = package_files('time_series_predictor/sklearn')

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="time_series_predictor",
    version="1.3.0",
    author="Daniel Kaminski de Souza",
    author_email="daniel@kryptonunite.com",
    description="Time Series Predictor",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://timeseriespredictor.readthedocs.io/",
    packages=['time_series_predictor'],
    package_data={'': extra_files},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'torch==1.5.0+cu92',
        'psutil==5.7.0',
        'tqdm==4.46.0',
        'skorch==0.8.0',
        'scipy==1.4.1'
    ],
    extras_require={
        'dev': [
            'pylint',
            'autopep8',
            'bumpversion',
            'twine',
            'python-dotenv',
            'python-dotenv[cli]',
            'lxml'
        ],
        'test': [
            'pytest>=4.6',
            'pytest-cov',
            'pandas',
            'seaborn',
            'sklearn',
            'python-dotenv',
            'lxml'
        ],
        'docs': [
            'sphinx',
            'rstcheck',
            'sphinx-autodoc-typehints',
            'nbsphinx',
            'recommonmark',
            'sphinx_rtd_theme',
            'pandas',
            'seaborn',
            'sklearn',
            'skorch',
            'jupyterlab',
            'matplotlib',
            'python-dotenv',
            'lxml'
        ]
    }
)
