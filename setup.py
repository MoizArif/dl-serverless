from setuptools import setup, find_packages

with open('LICENCE') as lfile:
    licence = lfile.read()

with open('README.md') as rfile:
    readme = rfile.read()

setup(
    name='osdel',
    version='1.0.0',
    licence=license,
    description='Optimized Serverless Deep Learning',
    long_description=readme,
    author='High Performance Distributed Systems Lab @RIT',
    author_email = '',
    packages=find_packages(exclude=('tests', 'docs'))
)
