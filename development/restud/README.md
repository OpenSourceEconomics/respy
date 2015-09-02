# Monte Carlo Evidence from Keane and Wolpin (1994)

The material in this directory allows to reproduce the Monte Carlo results reported in the two papers below:

> Michael P. Keane, Kenneth I. Wolpin (1994). The Solution and Estimation of Discrete Choice Dynamic Programming Models by Simulation and Interpolation: Monte Carlo Evidence. Federal Reserve Bank of Minneapolis, No. 181.

> Michael P. Keane, Kenneth I. Wolpin (1994). The Solution and Estimation of Discrete Choice Dynamic Programming Models by Simulation and Interpolation: Monte Carlo Evidence. The Review of Economics and Statistics, 76(4): 648-672.

### Codes

Source codes for the orginal FORTRAN program are available in the **codes** subdirectory. I post the original FORTRAN77 codes and also provide an upgraded FORTRAN95 version.

### Simulation

You can reproduce the Monte Carlo simulations simply by typing:
 	
    $ cd simulation; python create.py

This solves and simulates the model for the three parametrizations analyzed in the paper using the FORTRAN95 program and the ROBUPY package. The resulting choice probabilities summarized in **data.robupy.info** align between the implementations and the published results (see **original-results.pdf**). Minor discrepancies are due to randomness.  

