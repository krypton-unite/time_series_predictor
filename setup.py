"""
Setup
"""
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="time_series_predictor",
    version="0.0.1",
    author="Daniel Kaminski de Souza",
    author_email="daniel@kryptonunite.com",
    description="Time Series Predictor",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://timeseriespredictor.readthedocs.io/",
    packages=['time_series_predictor'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'torch',
        'psutil',
        'tqdm'
    ],
    extras_require={
        'dev': [
            'pylint',
            'autopep8',
            'sphinx',
            'rstcheck',
            'pytest',
            'pandas',
            'seaborn'
        ],
        'test': [
            'pytest',
            'pytest-cov',
            'pandas',
            'seaborn'
        ]
    }
)
