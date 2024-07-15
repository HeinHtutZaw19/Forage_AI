from setuptools import setup,find_packages
from typing import List

HYPHEN_E_DOT = '-e .'

def get_required_libraries(file:str)->List[str]:
    with open(file, 'r') as f:
        result = f.readlines()
        result = [e.replace('\n', '') for e in result]
        if(HYPHEN_E_DOT in result):
            result.remove(HYPHEN_E_DOT) 
    return result

setup(
    name="Cognizant Artificial Intelligence Job Simulation",
    author="Hein Htut Zaw",
    author_email="heinzaw.official@gmail.com",
    version="0.0.1",
    packages=find_packages(),
    install_requires=get_required_libraries('requirements.txt') 
)