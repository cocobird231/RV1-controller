import os
from glob import glob
from setuptools import setup

package_name = 'py_controller'

pkgFiles = [(os.path.join('lib', dp), [os.path.join(dp, f) for f in fn]) for dp, dn, fn in os.walk(os.path.join(package_name))]

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']), 
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'launch'), glob('launch/common.yaml')), 
        (os.path.join('share', package_name, 'launch'), glob('launch/service.json')), 
        *pkgFiles
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='coco',
    maintainer_email='cocobird231@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'run = py_controller.main:main', 
        ],
    },
)
