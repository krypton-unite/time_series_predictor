$script = $PSScriptRoot+"\delete_docs_build_folder.ps1"
& $script

docs/make html
$artifacts_folder = "docs/artifacts"
$destination_folder = "$artifacts_folder/html"
$script=$PSScriptRoot+"\delete_folder.ps1"
& $script -folder $destination_folder
mv docs/build/html $destination_folder -Force