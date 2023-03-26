from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(file_name:str) -> List[str]:
    '''
    Read the requirements file and return the list of requirements
    '''
    requirements = []
    with open(file_name) as f:
        for line in f:
            requirements.append(line.strip())

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements

setup(
    name='ML-Pipeline',
    version='0.0.1',
    author='Sandeep',
    author_email='sandeeppvn@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
)

