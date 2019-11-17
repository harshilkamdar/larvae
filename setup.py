"""
A setuptools based setup module.
See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

from setuptools import setup, find_packages
from os import path
from io import open

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='larvae',  # Required
    version='0.0.1',  # Required
    description='spectra in latent space',  
    author='Harshil Kamdar',  
    packages=['larvae'],
    package_dir={'larvae':'larvae'},
    install_requires=['numpy', 'matplotlib', 'sklearn', 'torch', 'umap'],  
)