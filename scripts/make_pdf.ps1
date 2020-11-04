$script = $PSScriptRoot+"\delete_docs_build_folder.ps1"
& $script

docs/make latex
docs/build/latex/make
mv docs/build/latex/timeseriespredictor.pdf docs/artifacts -Force