import re
from setuptools import setup, find_packages


def readme():
    """Return the contents of the project README file."""
    with open('README.md') as f:
        return f.read()


version = re.search(r"__version__ = ['\"]([^'\"]*)['\"]", open('statemodify/__init__.py').read(), re.M).group(1)

setup(
    name='statemodify',
    version=version,
    packages=find_packages(),
    url='https://github.com/IMMM-SFA/statemodify',
    license='BSD-2-Clause',
    author='',
    author_email='',
    description="A package to modify StateMod's input and output files for exploratory modeling",
    long_description=readme(),
    long_description_content_type="text/markdown",
    python_requires='>=3.8.*, <4',
    include_package_data=True,
    install_requires=[
        'numpy>=1.22.3',
        'pandas>=1.4.2',
        'joblib>=1.1.0',
        'SALib>1.4.5'
    ],
)
