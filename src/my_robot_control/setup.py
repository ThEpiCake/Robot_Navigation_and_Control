from setuptools import find_packages, setup
import os
from glob import glob
import sys

package_name = 'my_robot_control'

# Colcon/ament may pass flags not supported by some setuptools versions.
# Keep filtering minimal so ROS can still control install paths correctly.
_UNSUPPORTED_ARGS = {"--editable", "--uninstall"}
_UNSUPPORTED_WITH_VALUE = {"--build-directory"}

_filtered_argv = []
skip_next = False
for arg in sys.argv:
    if skip_next:
        skip_next = False
        continue
    if arg in _UNSUPPORTED_ARGS:
        continue
    if arg in _UNSUPPORTED_WITH_VALUE:
        skip_next = True
        continue
    _filtered_argv.append(arg)
sys.argv = _filtered_argv

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.py')),
        (os.path.join('lib', package_name),
            glob('scripts/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='thepicake',
    maintainer_email='etaybaro@post.bgu.ac.il',
    description='Part 1 dynamics, integrator, PID control and Gazebo playback.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'simulate_free  = my_robot_control.simulate_free:main',
            'simulate_pid   = my_robot_control.simulate_pid:main',
            'gazebo_control = my_robot_control.gazebo_control_node:main',
        ],
    },
)
