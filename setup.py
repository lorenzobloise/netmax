from setuptools import setup, find_packages

setup(
    name='netmax',
    packages=find_packages(include=['netmax', 'netmax.*']),
    version='0.1.0',
    description='Library for the problem of Influence Maximization in Social Networks',
    long_description='NetMax is a python framework that provides the implementation of several algorithms for the problem of Influence Maximization in Social Networks. It also addresses the problem of Competitive Influence Maximization as an extensive-form strategic game setting.',
    author='Lorenzo Bloise, Carmelo Gugliotta',
    author_email='l.bloise@outlook.it',
    install_requires=['networkx==3.3', 'numpy', 'tqdm', 'heapdict'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.12'
)