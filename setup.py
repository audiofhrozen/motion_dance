#!/usr/bin/env python
from distutils.version import LooseVersion
import os
import pip
from setuptools import find_packages
from setuptools import setup
import sys


if LooseVersion(sys.version) < LooseVersion('3.6'):
    raise RuntimeError(
        'DeepDancer requires Python>=3.6, '
        'but your Python is {}'.format(sys.version))
if LooseVersion(pip.__version__) < LooseVersion('19'):
    raise RuntimeError(
        'pip>=19.0.0 is required, but your pip is {}. '
        'Try again after "pip install -U pip"'.format(pip.__version__))

requirements = {
    'install': [
        'chainer>=4.5.0',
        'madmom>=0.14.0',
        'mir_eval>=0.3',
        'transforms3d>=0.3.0',
        'h5py>=2.7.0',
        'numpy>=1.10.0',
        'Soundfile>=0.10.1',
        'scipy>=0.19.1',
        'scikit-learn>=0.19.1',
        'pandas>=0.20.3',
        'python_speech_features',
        'chainerui',
        'scikit-learn',
        'BTET@git+https://github.com/Fhrozen/BTET.git',
        'python-osc',
    ],
    'setup': ['numpy', 'pytest-runner'],
    'test': [
        'pytest>=3.3.0',
        'pytest-pythonpath>=0.7.3',
        'hacking>=1.1.0',
        'mock>=2.0.0',
        'autopep8>=1.3.3']}
install_requires = requirements['install']
setup_requires = requirements['setup']
tests_require = requirements['test']
extras_require = {k: v for k, v in requirements.items()
                  if k not in ['install', 'setup']}

dirname = os.path.dirname(__file__)
setup(name='DeepDancer',
      version='0.1.0',
      url='http://github.com/Fhrozen/motion_dance',
      author='Nelson Yalta',
      author_email='nyalta21@gmail.com',
      description='Sequential Learning for Dance generation',
      long_description=open(os.path.join(dirname, 'README.md'),
                            encoding='utf-8').read(),
      license='Apache Software License',
      packages=find_packages(include=['deepdancer*']),
      install_requires=install_requires,
      setup_requires=setup_requires,
      tests_require=tests_require,
      extras_require=extras_require,
      classifiers=[
          'Programming Language :: Python',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3',
          'Development Status :: 5 - Production/Stable',
          'Intended Audience :: Science/Research',
          'Operating System :: POSIX :: Linux',
          'License :: OSI Approved :: Apache Software License',
          'Topic :: Software Development :: Libraries :: Python Modules'],
      )