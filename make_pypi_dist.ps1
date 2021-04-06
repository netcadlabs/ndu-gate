# pre-requires
# pip install wheel

$NDU_GATEWAY_MODULE_NAME = "ndu_gate_camera"

# clear old dist files
Get-ChildItem *.egg-info | foreach { Remove-Item -Recurse -Path $_.FullName }

If ((Test-Path "$NDU_GATEWAY_MODULE_NAME.egg-info") -eq $True){ Remove-Item -Recurse -Path "$NDU_GATEWAY_MODULE_NAME.egg-info" }
If ((Test-Path build) -eq $True){ Remove-Item -Recurse -Path build }
If ((Test-Path dist) -eq $True){ Remove-Item -Recurse -Path dist }

# preparing dist
python setup.py sdist bdist_wheel

# clear copied code
# Remove-Item -Recurse -Path ndu_gateway

# uploading dist
# python -m twine upload dist/*