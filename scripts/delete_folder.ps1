param (
    [string]$folder
 ) 
if (Test-Path $folder -PathType Container){
    rm $folder -r
}