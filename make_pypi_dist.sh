
rm -rf *.egg-info
rm -rf build
rm -rf dist

python3 setup.py sdist bdist_wheel

# preparing dist
# python3 -m twine upload dist/*

# uploading dist
# python3 -m twine upload dist/*

# installation
# python3 -m pip install --upgrade ndu_gate_camera

# installation without dependencies
# python3 -m pip install --upgrade --no-deps ndu_gate_camera