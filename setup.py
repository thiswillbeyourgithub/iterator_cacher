
from setuptools import setup, find_packages

setup(
    name="iterator_cacher",
    version="1.0.2",
    description="Decorate a python function to cache all subsets of iterable inputs",
    long_description="Decorate a python function to cache all subsets of iterable inputs",
    url="https://github.com/thiswillbeyourgithub/iterator_cacher",
    packages=find_packages(),

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    license="GPLv3",
    keywords=[],
    python_requires=">=3.9",

    entry_points={
        'console_scripts': [
            'iterator_cacher=iterator_cacher.__init__:cli_launcher',
        ],
    },

    install_requires=[
        "joblib >= 1.4.2",
        "beartype >= 0.19.0rc0",
    ],

)
