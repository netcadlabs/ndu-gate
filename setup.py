from setuptools import setup

from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

VERSION = "0.0.1"

setup(
    name='ndu-gate',
    version='0.0.1',
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
              'ndu_gate_camera.utility', 'ndu_gate_camera.dedectors',
              'ndu_gate_camera.dedectors.model', 'ndu_gate_camera.dedectors.vision',
              'ndu_gate_camera.dedectors.vision.nn', 'ndu_gate_camera.dedectors.vision.ssd',
              'ndu_gate_camera.dedectors.vision.ssd.config', 'ndu_gate_camera.dedectors.vision.utils',
              'ndu_gate_camera.dedectors.vision.datasets', 'ndu_gate_camera.dedectors.vision.transforms'],
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
