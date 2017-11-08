""" Setup script for motif. """
from setuptools import setup

import imp

version = imp.load_source('motif.version', 'motif/version.py')

if __name__ == "__main__":
    setup(
        name='motif',

        version=version.version,

        description='Melody object transcription framework',

        author='Rachel Bittner',

        author_email='rachel.bittner@nyu.edu',

        url='https://github.com/rabitt/motif',

        download_url='https://github.com/rabitt/motif/releases',

        packages=['motif'],

        package_data={'motif': []},

        long_description="""Melody object transcription framework""",

        keywords='music contour melody pitch',

        license='MIT',

        dependency_links=['git+ssh://git@github.com/librosa/librosa.git@0.5.0.dev0#egg=librosa-0.5.0.dev0'],

        install_requires=[
            'six',
            'numpy >= 1.8.0',
            'scipy >= 0.13.0',
            'scikit-learn >= 0.17.0',
            'matplotlib',
            'seaborn',
	    'librosa >= 0.5.0',
            'mir_eval >= 0.4.0',
            'sox'
        ],

        extras_require={
            'tests': [
                'mock',
                'pytest',
                'pytest-cov',
                'pytest-pep8',
            ],
            'docs': [
                'sphinx==1.2.3',  # autodoc was broken in 1.3.1
                'sphinxcontrib-napoleon',
                'sphinx_rtd_theme',
                'numpydoc',
            ],
        }
    )
