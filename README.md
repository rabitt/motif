# motif
Melodic Object TranscrIption Framework

[![Documentation Status](https://readthedocs.org/projects/motif/badge/?version=latest)](http://motif.readthedocs.io/en/latest/?badge=latest)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://raw.githubusercontent.com/rabitt/motif/master/LICENSE.md)
[![PyPI](https://img.shields.io/pypi/pyversions/Django.svg?maxAge=2592000)]()

[![Build Status](https://travis-ci.org/rabitt/motif.svg?branch=master)](https://travis-ci.org/rabitt/motif)
[![Coverage Status](https://coveralls.io/repos/github/rabitt/motif/badge.svg?branch=master)](https://coveralls.io/github/rabitt/motif?branch=master)


This library, inspired by [Uri Nieto's](https://github.com/urinieto) [msaf](https://github.com/urinieto/msaf), contains implementations of different melody transcription algorithms, broken down into independent modular components. This makes it easy to compare algorithms and test different combinations of components.

Algorithms are broken into the following components:

Contour Extractors
------------------
Methods that take an audio file as input and output all contour candidates.

Feature Extractors
------------------
Feature extraction on contour observations.

Contour Classifiers
-------------------
Methods that assign scores to contours.

Contour Decoders
----------------
Methods that take contour candidates and scores and produce single f0 outputs.


Documentation
=============
Documentation can be found [here](http://motif.readthedocs.io).


See Also
========
[msaf](https://github.com/urinieto/msaf): Music Structure Analysis Framework

[A Comparison of Melody Extraction Methods Based on Source-Filter Modelling](http://www.justinsalamon.com/uploads/4/3/9/4/4394963/bosch_melodyextraction_ismir2016.pdf)
Bosch, J.J., Bittner, R.M., Salamon, J. and Gómez Gutiérrez, E.
17th International Society for Music Information Retrieval (ISMIR) conference


