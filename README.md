# FIESTA

*Authors: [Jiten Dhandha](mailto:jitendhandha@gmail.com), [Zoe Faes](mailto:zoe.faes@esa.int), [Rowan J. Smith](mailto:rjs22@st-andrews.ac.uk)*

![version](https://img.shields.io/badge/version-1.0.0-blue)
[![Documentation Status](https://readthedocs.org/projects/fiesta-astro/badge/?version=latest)](https://fiesta-astro.readthedocs.io/en/latest/?badge=latest)
[![arXiv](https://img.shields.io/badge/arXiv-2307.12428-b31b1b.svg)](https://arxiv.org/abs/2307.12428)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8176097.svg)](https://doi.org/10.5281/zenodo.8176097)

***FIESTA*** stands for **FI**lamentary **ST**ructure
**A**nalysis, and is an astrophysical toolkit for studying filamentary
networks in density fields defined on unstructured meshes. It is
primarily built for use with the multi-physics hydrodynamical code *AREPO*
([Springel, 2010](https://doi.org/10.1111/j.1365-2966.2009.15715.x))
and the filament identification tool *DisPerSE* ([Sousbie,
2011](https://doi.org/10.1111/j.1365-2966.2011.18394.x)), but can
also be used more generally with other softwares through some
modifications. It includes tools such as:

1. reading and writing data
files for the aforementioned softwares;

1. 2D and 3D visualizations of
simulations and filamentary networks, along with functions for
statistical analysis;

1. algorithms for characterizing filament
properties such as length, mass, density profile, and more!

## Usage and license

The code initially started as part of the Masters project “*Beads on a
string: Connecting Turbulence to Massive Star Formation*” at the University of Manchester
by Jiten Dhandha and Zoe Faes under the supervision of Rowan J. Smith.

The code is released under GNU General Public License v3.0, 
which means it can be freely used, shared, modified and distributed. For more details on
the filament characterization algorithms, please see the associated paper 
[Dhandha, Faes & Smith (2023)](https://arxiv.org/abs/2307.12428) where the code was first introduced.
If you use the code in your own publication, please cite Dhandha, Faes & Smith (2023).
The paper is also accompanied by a data release on [Zenodo](https://doi.org/10.5281/zenodo.8176097),
the analysis of which led to the development of this code.

## Installation

***FIESTA***  requires the following packages to be installed:
-  *numpy*
-  *scipy*
-  *matplotlib*
-  *astropy*

The code can easily installed from the GitHub repository, by first cloning the repository 
or downloading the zip file, and then running the following command in the top level directory:

```
pip install .
```

## Bugs and suggestions

If you find any bugs, have suggestions or requests, or want help
with the toolkit, please feel free to send one of the authors an email
or raise an issue in the GitHub repository. We would be delighted to
talk about it!