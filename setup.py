from setuptools import setup,find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()


setup(
    name="Hotel-Reservations-Prediction",
    version="0.1",
    author= "Nasir Hussain",
    packages= find_packages(),
    install_requires = requirements,
)