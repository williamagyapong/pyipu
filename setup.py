"""Setup script for the PyIPU package."""

import os
from setuptools import setup, find_packages

# Get version from version.py
version_file = os.path.join(os.path.dirname(__file__), 'pyipu', 'version.py')
with open(version_file, 'r') as f:
    exec(f.read())

# Read the long description from README.md
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='pyipu',
    version=__version__,
    description='Python Implementation of Iterative Proportional Updating',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='William O. Agyapong',
    author_email='williamofosuagyapong@gmail.com',
    url='https://github.com/williamagyapong/pyipu',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.19.0',
        'pandas>=1.0.0',
        'matplotlib>=3.0.0',
        'scikit-learn>=0.24.0',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering',
    ],
    python_requires='>=3.7',
)
