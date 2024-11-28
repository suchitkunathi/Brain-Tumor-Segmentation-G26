from typing import List
from setuptools import setup, find_packages

HYPHEN_E_DOT = '-e .'

def get_requirements(filepath: str) -> List[str]:
    requirements = []
    with open(filepath) as fileobj:
        requirements = fileobj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
        
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements

setup (
    name="brain_tumor",
    version="0.0.0",
    author="suchit",
    author_email="suchit.kunathi@gmail.com",
    packages=find_packages(),
    install_requires = get_requirements('requirements.txt')
)

 
