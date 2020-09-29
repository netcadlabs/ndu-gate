from setuptools import setup

from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

VERSION = "0.0.3"

setup(
    name='ndu-gate',
    version='0.0.3',
    license='',
    author='netcadlabs',
    author_email='netcadinnovationlabs@gmail.com',
    description='NDU Gate Camera Service',
    long_description=long_description,
    long_description_content_type="text/markdown",
    include_package_data=True,
    python_requires=">=3.5",
    packages=['ndu_gate_camera', 'ndu_gate_camera.api', 'ndu_gate_camera.camera',
              'ndu_gate_camera.camera.video_sources', 'ndu_gate_camera.camera.result_handlers',
              'ndu_gate_camera.utility', 'ndu_gate_camera.detectors',
              'ndu_gate_camera.detectors.model', 'ndu_gate_camera.detectors.vision',
              'ndu_gate_camera.detectors.vision.nn', 'ndu_gate_camera.detectors.vision.ssd',
              'ndu_gate_camera.detectors.vision.ssd.config', 'ndu_gate_camera.detectors.vision.utils',
              'ndu_gate_camera.detectors.vision.datasets', 'ndu_gate_camera.detectors.vision.transforms'],
    url='https://github.com/netcadlabs/ndu-gate',
    download_url='https://github.com/netcadlabs/ndu-gate/archive/%s.tar.gz' % VERSION,
    install_requires=[
        'pip',
        'gdown',
        'PyYAML',
        'simplejson'
    ],
    package_data={
        "*": ["config/*"]
    },
    entry_points={
        'console_scripts': [
            'ndu-gate = ndu_gate_camera.ndu_camera:daemon'
        ]},
)
