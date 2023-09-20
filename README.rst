Overview
========
|docs| |version| |doi|

Python interface for the Swarm Ionospheric Polar Electrodynamics (SWIPE) model.

The SWIPE model is an empirical model of high-latitude ionospheric electrodynamics, and is a combination of the Average Magnetic field and Polar current System (AMPS) model and the Swarm High-latitude Convection (Swarm Hi-C) model.

** AMPS
The AMPS model magnetic field and currents are continuous functions of solar wind velocity, the interplanetary magnetic field, the tilt of the Earth's dipole magnetic field with respect to the Sun, and the 10.7 cm solar radio flux index F10.7. Given these parameters, model values of the ionospheric magnetic field can be calculated anywhere in space, and, with certain assumptions, on ground. The full current system, horizontal + field-aligned, are defined everywhere in the polar regions. The model is based on magnetic field measurements from the low Earth orbiting Swarm and CHAMP satellites.

** Swarm Hi-C
The Swarm Hi-C model high-latitude ionospheric convection is a function of the same input parameters used for the AMPS model. Given these parameters, model values of the high-latitude ionospheric convection, potential, and electric field can be calculated. The model is based on ion drift measurements from Swarm A and Swarm C.

pyswipe can be used to calculate and plot several different quantities on a grid. The parameters that are available for calculation/plotting are:

- electric potential (scalar)
- electric field E (vector)
- convection v = - cross(E, B) (vector)
- height-integrated electromagnetic work = dot(J, E) (scalar) in the earth's rotating frame of reference, with J given by the AMPS model and E by Swarm Hi-C
- Hall and Pedersen conductances (scalars)
- Poynting flux (scalar)

For questions and comments, please contact spencer.hatch at ift.uib.no

Installation
------------

Using pip::

    pip install pyswipe


Dependencies:

- numpy
- pandas
- dask
- matplotlib (with LaTeX support, see https://matplotlib.org/users/usetex.html)
- scipy (scipy.interpolate for plotting purposes)
- apexpy (magnetic coordinate conversion)

Quick Start
-----------
.. code-block:: python

    >>> # initialize by supplying a set of external conditions:
    >>> from pyswipe import SWIPE
    >>> m = SWIPE(350, # Solar wind velocity in km/s 
                  -4, # IMF By (GSM) in nT
                  -3, # IMF Bz (GSM) in nT, 
                  20, # dipole tilt angle in degrees 
                  80) # F107_index
    >>> # make summary plot:
    >>> m.plot_potential()

.. image:: docs/static/example_plot.png
    :alt: Field-aligned (colour) and horizontal (pins) currents
    
.. code-block:: python

    >>> # All the different current functions can be calculated on
    >>> # a pre-defined or user-specified grid.
    >>> import numpy as np 
    >>> mlat, mlt = np.array([75, -75]), np.array([12, 12])
    >>> Ju = m.get_upward_current(mlat, mlt)
    >>> Ju
    array([ 0.23323377, -0.05599236])

Documentation
-------------
IF this were pyamps, we could point to `http://pyamps.readthedocs.io` . But it's not!

References
----------
Laundal, K. M., Finlay, C. C., Olsen, N. & Reistad, J. P. (2018), Solar wind and seasonal influence on ionospheric currents from Swarm and CHAMP measurements, Journal of Geophysical Research - Space Physics. `doi:10.1029/2018JA025387 <https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2018JA025387>`_

*pyAMPS uses an updated set of model coefficients compared to the model discussed in the paper. You can use pyAMPS and the scripts in pyamps/climatology_plots/ to produce updated versions of Figures 5-7 and 9-11 from the paper*

See also:
Laundal, K. M., Finlay, C. C. & Olsen, N. (2016), Sunlight effects on the 3D polar current system determined from low Earth orbit measurements. Earth Planets Space. `doi:10.1186/s40623-016-0518-x <https://earth-planets-space.springeropen.com/articles/10.1186/s40623-016-0518-x>`_ 


Acknowledgments
---------------
The code is produced with support from ESA through the Swarm Data Innovation and Science Cluster (Swarm DISC). For more information on Swarm DISC, please visit https://earth.esa.int/web/guest/missions/esa-eo-missions/swarm/disc


Badges
------

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - |requires| 
    * - package
      - | |version|

.. |docs| image:: https://readthedocs.org/projects/pyswipe/badge/?version=latest
    :target: http://pyswipe.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. |version| image:: https://badge.fury.io/py/pyswipe.svg
    :alt: PyPI Package latest release
    :target: https://badge.fury.io/py/pyswipe

.. |coveralls| image:: https://coveralls.io/repos/github/dartspacephysiker/pyswipe/badge.svg
    :target: https://coveralls.io/github/dartspacephysiker/pyswipe
    :alt: Coverage Status

.. |requires| image:: https://requires.io/github/dartspacephysiker/pyswipe/requirements.svg?branch=master
    :target: https://requires.io/github/dartspacephysiker/pyswipe/requirements/?branch=master
    :alt: Requirements Status

.. |doi| image:: https://zenodo.org/badge/DOI/
    :target: https://doi.org/
