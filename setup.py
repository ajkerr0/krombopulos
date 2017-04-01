
from setuptools import find_packages, setup

setup(name="puddy",
      version="0.1.0",
      description="A set of machine learning tools.",
      author="Alex Kerr",
      author_email="ajkerr0@gmail.com",
      url="https://github.com/ajkerr0/puddy",
      packages=find_packages(),
      install_requires=[
      'numpy', 'scipy', 'matplotlib'
      ],
      )
