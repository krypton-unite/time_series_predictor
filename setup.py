"""
Setup
"""
import distutils.cmd
import distutils.log
import os
import subprocess

import setuptools


class UpgradeCommand(distutils.cmd.Command):
    """A custom command to run pip-compile generating hashes and outputting to
    requirements-lock.txt."""

    description = 'update requirements-lock.txt with upgraded packages'
    user_options = [
        # The format is (long option, short option, description).
        ('output-file=', None, 'path to output requirements file'),
    ]

    def initialize_options(self):
        """Set default values for options."""
        # Each user option must be listed here with their default value.
        self.output_file = 'requirements-lock.txt'

    def finalize_options(self):
        """Post-process options."""
        if self.output_file:
            assert os.path.exists(self.output_file), (
                'Output file %s does not exist.' % self.output_file)

    def run(self):
        """Run command."""
        command = ['pip-compile']
        if self.output_file:
            # pylint: disable=line-too-long
            command.append(
                '--find-links=https://download.pytorch.org/whl/torch_stable.html --generate-hashes --upgrade --output-file=%s' % self.output_file)
        command.append(os.getcwd())
        print(command)
        self.announce(
            'Running command: %s' % str(command),
            level=distutils.log.INFO)
        subprocess.check_call(command)


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
    cmdclass={
        'upgrade': UpgradeCommand,
    },
    name="time_series_predictor",
    version="1.3.1",
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
        'skorch',
        'scipy==1.4.1',  # via skorch
        'psutil'
    ],
    extras_require={
        'dev': [
            'wheel',
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
            'lxml',
            'requests'
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
            'jupyterlab',
            'matplotlib',
            'python-dotenv',
            'lxml'
        ]
    }
)
