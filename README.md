# DMQMC-GPR
Code for applying Gaussian process regression to DMQMC data from HANDE-QMC.

## Python packages
The initial implementation was developed with the following packages
and corresponding versions. This is not intended to be comprehensive
but instead is focused on those required by GPy as indicated by pip.
Additionally, packages related to running the example and test scripts.

0. Python 3.9.12
1. GPy 1.10.0
2. paramz 0.9.5
3. numpy 1.21.5
4. scipy 1.7.3
5. six 1.16.0
6. Cython 0.29.28
7. matplotlib 3.5.1
8. pandas 1.4.1

## Installation and running
For installation it is recommended to use anaconda or miniforge. The required
libraries corresponding to the original implementation are listed within the
`conda_env.yml` file. To make an install using anaconda we can run the following
commands:
1. `conda env create -f conda_env.yml`
2. `conda activate dmqmcgpr`
3. `export PYTHONPATH=$(pwd)/qmcgpr/:$PYTHONPATH`

Note the final step is only required to import the library without appending
the path (see `qmcgpr/examples` for more information).

After following the first two steps above using anaconda 24.1.2, we are able to
run the simple test suite from the `qmcgpr/tests` folder using the command:
1. `export OMP_NUM_THREADS=1`
2. `bash ./runtests.sh *simple.py`

Note that a more extensive test suite can be run with the following command:
1. `bash ./runtests.sh`

However our testing has shown that the results for the tests (and the GPR
code used within) are hardware, library, and thread count dependent.
Additionally, there are likely several factors which are yet to be
discovered that can impact the results.

As such, the simple tests are meant to serve as a way to check that the code
is functioning as a whole. As well as demonstrate how one can take steps to
ensure that the relevant data are reproduced at a late date. That is to say
it is important to store the original data set used to apply the code, as well
as the output for the code in the form of the parameter files etc.

For the full test suite, these were run and fully passed on an intel based
2016 MacBook Pro from which the data were originally produced.

The simple tests are able to reproduce the data relevant to figures within
the manuscript (see link below) on a MacBook Pro M3 using a single OMP thread.

## Citation
For referencing the original work this repository is based on use:

    @article{vanbenschoten_electronic_2023,
        author = {Van Benschoten, William Z. and Weiler, Laura and Smith, Gabriel J. and Man, Songhang and DeMello, Taylor and Shepherd, James J.},
        title = {Electronic specific heat capacities and entropies from density matrix quantum Monte Carlo using Gaussian process regression to find gradients of noisy data},
        journal = {The Journal of Chemical Physics},
        volume = {158},
        number = {21},
        pages = {214115},
        year = {2023},
        month = {06},
        issn = {0021-9606},
        doi = {10.1063/5.0150702},
        url = {https://doi.org/10.1063/5.0150702},
    }

## Useful links
*   The GPy github page:
    -   https://github.com/SheffieldML/GPy
*   The HANDE-QMC github page:
    -   https://github.com/hande-qmc/hande
*   The original manuscript this repository was generated from:
    -   https://doi.org/10.1063/5.0150702

## Funding
*   Research was supported by the U.S. Department of Energy, Office of Science,
    Office of Basic Energy Sciences Early Career Research Program (ECRP) under
    Award Number DE-SC0021317. 
