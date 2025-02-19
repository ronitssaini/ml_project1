from setuptools import setup, find_packages

def get_requirements(file_path):
    with open(file_path, 'r') as file:
        return file.read().splitlines()

setup(
    name='ml_project1',
    version='0.0.1',
    description='My Python package',
    author='Ronit Saini',
    author_email='ronittsaini@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
    )