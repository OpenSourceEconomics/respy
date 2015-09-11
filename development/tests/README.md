# Development Tests

This directory contains the main testing infrastructure for the *robupy* package


### Codes

Source codes for the original FORTRAN77 and an upgraded FORTRAN95 program are available in the **codes** subdirectory.

### Simulation

You can generate their simulated samples by simply typing:
 	
    $ cd simulation; python create.py

This solves and simulates the model for the three parametrizations analyzed in the paper using their program and the robupy package. The resulting choice probabilities summarized in **data.robupy.info** align between the implementations and the published results (see **original-results.pdf**). Minor discrepancies are due to randomness.  

