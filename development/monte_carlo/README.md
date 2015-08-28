# Monte Carlo Evidence from Keane and Wolpin (1994)

The material in this directory allows to reproduce the Monte Carlo results reported in the two papers below:

> Michael P. Keane, Kenneth I. Wolpin (1994). The Solution and Estimation of Discrete Choice Dynamic Programming Models by Simulation and Interpolation: Monte Carlo Evidence. Federal Reserve Bank of Minneapolis, No. 181.

> Michael P. Keane, Kenneth I. Wolpin (1994). The Solution and Estimation of Discrete Choice Dynamic Programming Models by Simulation and Interpolation: Monte Carlo Evidence. The Review of Economics and Statistics, 76(4): 648-672.

#### Execution
 
    $ python create.py

The resulting choice probabilities reported in **data.robupy.info** align with their results provided in **original-results.pdf**. Minor discrepancies are due to randomness.  

### Original

The original source codes are available in the **sources** subdirectory. An upgraded **FORTRAN95** implementation is available [here](https://github.com/robustToolbox/package/tree/master/development/monte_carlo/original_codes/f95).
