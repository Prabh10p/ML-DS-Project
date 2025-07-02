from setuptools import find_packages, setup
from typing import List

def get_requirements(file_path: str) -> List[str]:
    with open(file_path) as obj_file:
        requirements = obj_file.readlines()
        requirements = [i.strip() for i in requirements]
        if "-e ." in requirements:
            requirements.remove("-e .")
    return requirements

setup(
    name='ML_DS_Project',  # Avoid using slashes in names
    version='0.0.1',
    author='Prabhjot Singh',
    author_email='prabh2004p@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
