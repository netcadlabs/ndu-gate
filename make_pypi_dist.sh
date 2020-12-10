
rm -rf *.egg-info
rm -rf build
rm -rf dist



python3 setup.py sdist bdist_wheel

# preparing distß
# python3 -m twine upload dist/*

# installation
# python3 -m pip install --upgrade ndu_gate_camera