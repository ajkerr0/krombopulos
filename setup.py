
from setuptools import find_packages, setup

setup(name="krombopulos",
      version="0.1.0",
      description="A set of optimization and machine learning tools.",
      author="Alex Kerr",
      author_email="ajkerr0@gmail.com",
      url="https://github.com/ajkerr0/krombopulos",
      packages=find_packages(),
      install_requires=[
      'numpy', 'scipy', 'matplotlib'
      ],
      )
