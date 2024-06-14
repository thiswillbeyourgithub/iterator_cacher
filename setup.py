
from setuptools import setup, find_packages
from setuptools.command.install import install

setup(
    name="iterator_cacher",
    version="0.0.1",
    description="decorator to transparently cache each element of an iterable",
    long_description="decorator to transparently cache each element of an iterable",
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
        "dill >= 0.3.8",
        "joblib >= 1.4.2",
        "typeguard >= 4.3.0"
        # TODO_req
    ],
    extra_require={
    'optionnal_feature': [
        # TODO_req
        ]
    },

)
