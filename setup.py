""" Setup script for sox. """
from setuptools import setup

if __name__ == "__main__":
    setup(
        name='motif',

        version='0.0.1',

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

        install_requires=[
            'numpy',
            'cesium',
            'librosa',
            'mir_eval',
            'sklearn'
        ],

        extras_require={
            'tests': [
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
