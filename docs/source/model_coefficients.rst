Coefficient file format
-----------------------
The coefficients used in the model can be found in `pyswipe/coefficients/SW_OPER_MIO_SHA_2E_00000000T000000_99999999T999999_0101.txt`.

Below is a description of the coefficient file. This information is not needed to use pyAMPS, but it can be useful if you want to do something more advanced and write new code or modify existing code.

The coefficient file is a .txt file with fixed width columns. It can be read with the Python pandas library by the following code: 

.. code-block:: python

    >>> coeff_fn = 'SW_OPER_MIO_SHA_2E_00000000T000000_99999999T999999_0101.txt'
    >>> names = ([x for x in open(coeff_fn).readlines() if x.startswith('#')][-1][1:]).strip().split(' ') 
    >>> coeffs = pd.read_table(coeff_fn, skipinitialspace = True, comment = '#', sep = ' ', names = names, index_col = [0, 1])

The first two columns contain the spherical harmonic wave numbers n and m. The other 76 columns are named

| ``tor_c_<param>``
| ``tor_s_<param>``
| ``pol_c_<param>``
| ``pol_s_<param>``

where ``<param>`` refers to the 19 external parameters: ``const``, ``sinca``, ``epsilon_cosca``, ``epsilon_sinca``, ``epsilon_cosca``, ``tilt``, ``tilt_sinca``, ``tilt_cosca``, ``tilt_epsilon``, ``tilt_epsilon_sinca``, ``tilt_epsilon_cosca``, ``tau``, ``tau_sinca``, ``tau_cosca``, ``tilt_tau``, ``tilt_tau_sinca``, ``tilt_tau_cosca``, or ``f107``.

Here, ``const`` refers to a constant term. ``sinca`` and ``cosca`` are *sin(IMF clock angle)* and *cos(IMF clock angle)*, respectively. ``epsilon`` is the Newell coupling function divided by 1000 with inputs in nT and km/s. ``tau`` is the Newell coupling function with *sin* replaced by *cos*. ``f107`` is the F10.7 index in standard units (sfu). ``tilt`` is the dipole tilt angle in degrees. Underscores in this list refer to multiplication, e.g.: ``tilt_tau`` is the dipole tilt angle multiplied by tau. 

The prefix of the column names (apart from ``n`` and ``m``) denote the term in a real expansion of the poloidal and toroidal parts of the ionospheric magnetic field as described in Laundal et al. (2016). ``tor`` and ``pol`` refer to toroidal and poloidal, respectively. ``_c`` and ``_s`` refer to the *cos* and *sin* terms, respectively.

To get the spherical harmonic coefficients for a given set of external parameters, the coefficients should be multiplied by the corresponding external parameter (1 in the case of ``const``), and summed so that four columns remain, one for each ``tor_c``, ``tor_s``, ``pol_c``, and ``pol_s``. 

There are several missing entries: The *(n, 0)* terms for the ``_s`` coefficients are undefined, since *sin(m\*x) = 0* for all *x* if *m = 0*. In addition, the toroidal expansion have more terms (higher spatial resolution) than the poloidal expansion, so that ``pol_c_<param>`` and ``pol_s_<param>`` are missing/undefined for *n > 45*.
