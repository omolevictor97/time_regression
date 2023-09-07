from setuptools import find_packages, setup
from typing import List


## We write a custom function to install packages

def get_requirements(file_path:str) -> List[str]:
    try:
        requirements = []
        with open(file_path, "r") as file:
            requirements = file.readlines()
            requirements = [req.replace("\n", "") for req in requirements if req != "-e ."]
        return requirements
    except Exception as e:
        return str(e)


setup(
    name="Time Regress",
    version="0.0.1",
    author="Oshionwu Victor",
    author_email= "victoropeyemi97@outlook.com",
    packages= find_packages(),
    install_packages = get_requirements("requirements.txt")
)