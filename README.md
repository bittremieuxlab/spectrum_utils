# spectrum_utils

[![conda](https://img.shields.io/conda/vn/bioconda/spectrum_utils?color=green)](https://bioconda.github.io/recipes/spectrum_utils/README.html)
[![PyPI](https://img.shields.io/pypi/v/spectrum_utils?color=green)](https://pypi.org/project/spectrum_utils/)
[![Tests](https://github.com/bittremieux/spectrum_utils/actions/workflows/tests.yml/badge.svg)](https://github.com/bittremieux/spectrum_utils/actions/workflows/tests.yml)
[![Docs](https://github.com/bittremieux/spectrum_utils/actions/workflows/docs.yml/badge.svg)](https://github.com/bittremieux/spectrum_utils/actions/workflows/docs.yml)
[![Documentation](https://readthedocs.org/projects/spectrum-utils/badge/?version=latest)](https://spectrum-utils.readthedocs.io/en/latest/?badge=latest)

spectrum_utils is a Python package for efficient mass spectrometry data processing and visualization.

spectrum_utils contains the following features:

- Spectrum loading from online proteomics and metabolomics data resources using the [Universal Spectrum Identifier (USI)](https://www.psidev.info/usi) mechanism.
- Common spectrum processing operations (precursor & noise peak removal, intensity filtering, intensity scaling) optimized for computational efficiency.
- Annotating observed spectrum fragments using the [ProForma 2.0 specification](https://www.psidev.info/proforma) for (modified) peptidoforms.
- Publication-quality, fully customizable spectrum plotting and interactive spectrum plotting.

![spectrum_utils logo](spectrum_utils.png)

## Installation

spectrum_utils requires Python version 3.10+ and can be installed with pip or conda.

Using pip:

    pip install spectrum_utils[iplot]

Using conda:

    conda install -c bioconda spectrum_utils

## Documentation

Please see the [documentation](https://spectrum-utils.readthedocs.io/) for detailed installation instructions, usage examples, the API reference, and more information.

## Citation
 
spectrum_utils is freely available as open source under the [Apache 2.0 license](http://opensource.org/licenses/Apache-2.0).

When using spectrum_utils, please cite the following manuscripts:
 
- Wout Bittremieux. "spectrum_utils: A Python package for mass spectrometry data processing and visualization." _Analytical Chemistry_ **92**, 659--661 (2020) doi:[10.1021/acs.analchem.9b04884](https://doi.org/10.1021/acs.analchem.9b04884).
- Wout Bittremieux, Lev Levitsky, Matteo Pilz, Timo Sachsenberg, Florian Huber, Mingxun Wang, Pieter C. Dorrestein. "Unified and standardized mass spectrometry data processing in Python using spectrum_utils" _Journal of Proteome Research_ **22**, 625--631 (2023) doi:[10.1021/acs.jproteome.2c00632](https://doi.org/10.1021/acs.jproteome.2c00632).
