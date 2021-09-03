import setuptools
from molx.version import __version__

with open("README.md", "r") as fh:
    long_description = fh.read()

setup_requires = ['pytest-runner']
# tests_require = ['pytest', 'pytest-cov', 'mock']

setuptools.setup(
    name="moleculex",
    version=__version__,
    author="DIVE Lab@TAMU",
    author_email="sji@tamu.edu",
    maintainer="DIVE Lab@TAMU",
    license="GPLv3",
    description="MoleculeX: a new and rapidly growing suite of machine \
    learning methods and software tools for molecule exploration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/divelab/MoleculeX",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    install_requires=['scipy',
                      'cilog',
                      'typed-argument-parser==1.5.4',
                      'captum==0.2.0',
                      'shap',
                      'IPython',
                      'tqdm',
                      'rdkit-pypi',
                      'pandas'],
    python_requires='>=3.6',
    setup_requires=setup_requires,
#     tests_require=tests_require,
#     extras_require={'test': tests_require},
#     include_package_data=True
)
