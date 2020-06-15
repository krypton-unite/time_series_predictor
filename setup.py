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
    version="1.1.2",
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
        'torch',
        'psutil',
        'tqdm',
        'skorch',
        'scipy'
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
            'sklearn'
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
            'jupyterlab',
            'matplotlib'
        ]
    }
)
