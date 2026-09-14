import os
from glob import glob

from setuptools import setup

package_name = "semantic_objects"

setup(
    name=package_name,
    version="0.2.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages",
         ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="mic-711",
    maintainer_email="chathatpol@gmail.com",
    description="Camera + 2D lidar semantic landmark fusion",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "semantic_objects_node = semantic_objects.semantic_objects_node:main",
        ],
    },
)
