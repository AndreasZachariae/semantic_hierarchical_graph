import os
from glob import glob
from setuptools import setup

package_name = 'shg'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('lib', package_name, 'semantic_hierarchical_graph'), glob('../semantic_hierarchical_graph/*.py')),
        (os.path.join('lib', package_name, 'semantic_hierarchical_graph/types'),
         glob('../semantic_hierarchical_graph/types/*.py')),
        (os.path.join('lib', package_name, 'semantic_hierarchical_graph/planners'),
         glob('../semantic_hierarchical_graph/planners/*.py')),
        (os.path.join('lib', package_name, 'path_planner_suite'), glob('../path_planner_suite/*.py')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        # (os.path.join('share', package_name, 'config'), glob('config/*.yaml'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Andreas Zachariae',
    maintainer_email='andreas.zachariae@gmail.com',
    description='Semantic Hierachical Graph for multi-story Navigation',
    license='Creative Commons Attribution-NonCommercial 4.0 International License',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'graph_node = shg.graph_node:main'
        ],
    },
)
