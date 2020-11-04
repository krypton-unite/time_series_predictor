$script = $PSScriptRoot+"\delete_docs_build_folder.ps1"
& $script

docs/make epub
mv docs/build/epub/TimeSeriesPredictor.epub docs/artifacts -Force