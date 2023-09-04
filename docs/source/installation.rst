Installation
============



The easiest way to install pyswipe is using ``pip``::

    pip install pyswipe

pyswipe has the following dependencies:

- numpy
- dask
- matplotlib
- scipy (scipy.interpolate required for plotting purposes)
- pandas (for reading csv file containing model coefficients)
- apexpy (magnetic coordinate conversion)



pyswipe can also be installed directly from source. You will then manually have to install the relevant dependencies. The source code can then be downloaded from Github and installed::

    git clone https://github.com/dartspacephysiker/pyswipe
    cd pyswipe
    python setup.py install
