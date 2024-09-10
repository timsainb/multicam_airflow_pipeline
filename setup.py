from setuptools import setup, find_packages

def read_requirements():
    with open('requirements.txt') as f:
        return f.read().splitlines()

print("in setup.py")

setup(
    name='multicamera-airflow-pipeline',
    version='0.1.0',
    packages=find_packages(),
    install_requires=read_requirements(),
)
