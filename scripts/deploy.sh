#!/bin/sh
python setup.py sdist bdist_wheel
version="3.0.7"
files_to_handle_str="dist/time_series_predictor-$version*" 
twine check $files_to_handle_str
twine upload $files_to_handle_str