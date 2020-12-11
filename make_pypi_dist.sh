
rm -rf *.egg-info
rm -rf build
rm -rf dist

# preparing dist
python3 setup.py sdist bdist_wheel


# uploading dist
# python3 -m twine upload dist/*