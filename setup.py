from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

__author__ = 'Lorenzo Bloise, Carmelo Gugliotta'
__license__ = 'MIT License'
__email__ = 'l.bloise@outlook.it - carmelo.gugliotta00@gmail.com'

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='netmax',
    packages=find_packages(include=['netmax', 'netmax.*']),
    version='0.1.0',
    description='Library for the problem of Influence Maximization in Social Networks',
    long_description=long_description,
    author=__author__,
    author_email=__email__,
    license=__license__,
    install_requires=['networkx==3.3', 'numpy', 'tqdm', 'heapdict'],
    classifiers=[
        'Programming Language :: Python :: 3',
        f'License :: OSI Approved :: {__license__}',
        'Operating System :: OS Independent',
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Network Analysis'
    ],
    keywords='influence-maximization network-analysis simulator complex-networks',
    python_requires='>=3.12',
)