FIESTA
=====================

.. toctree::
   :maxdepth: 3
   :hidden:

   docu

**Version**: |version|

**FIESTA** stands for **FI**\ lam\ **E**\ ntary **ST**\ ructure **A**\ nalysis, and is an astrophysical toolkit for studying filamentary networks in density fields defined on unstructured meshes. It is primarily built for use with the multi-physics hydrodynamical code AREPO (`Springel, 2010 <https://doi.org/10.1111/j.1365-2966.2009.15715.x>`_) and the filament identification tool DisPerSE (`Sousbie, 2011 <https://doi.org/10.1111/j.1365-2966.2011.18394.x>`_), but can also be used more generally with other softwares through some modifications. It includes tools such as:

1. reading and writing data files for the aforementioned softwares; 
2. 2D and 3D visualizations of simulations and filamentary networks, along with functions for statistical analysis;
3. algorithms for characterizing filament properties such as length, mass, density profile, and more!

This code initially started as part of the Masters project "*Beads on a string: Connecting Turbulence to Massive Star Formation*" at the Jodrell Bank Center for Astrophysics (JBCA), University of Manchester by Jiten Dhandha (jitendhandha@gmail.com) and Zoe Faes (zoe.faes@esa.int) under the supervision of Dr. Rowan Smith (rowan.smith@manchester.ac.uk). It has been developed over the course of a year, almost entirely from scratch, so we hope it finds some use in the astrophysics community! :\)

Installation
----------------------

FIESTA can be easily installed or upgraded using the following pip commands \(NON FUNCTIONAL CURRENTLY\)::
   
   pip install fiesta
   pip install --upgrade fiesta

To install directly from the GitHub repository, download the zip file and run the following command in the top level directory::
   
   pip install .

This might be useful for those wanting to modify the source code or add their own functionality locally!

AHHH... bugs!!
----------------------

If you have find any bugs, have suggestions or requests, or want help with the toolkit, please feel free to send one of the authors a message or raise an issue in the GitHub repository. We would be delighted to talk about it!

License and citing
----------------------

The code is released under GNU General Public License v3.0, which means it can be freely used, shared, modified and distributed. If you use the code for a publication, please cite it as "Faes et al. (in prep)".