python setup.py sdist bdist_wheel
$version="1.4.5"
$files_to_handle_str="dist/time_series_predictor-$version*" 
twine check $files_to_handle_str
twine upload $files_to_handle_str