# Testing Infrastructure

This directory contains the main testing infrastructure for the ***RobuPy*** package.


### Random

The setup allows to test the ***RobuPy***  package for a fixed amount of time. Random model specifications are generated and the integrity of the package is checked. The tests are available [here](https://github.com/robustToolbox/package/blob/master/development/tests/random/modules/battery.py).

For more details on how to run the tests:

    $ ./run -h

### Fixed

This setup runs battery of tests for fixed model specifications and allows to ensure that the results of the package does not change during refactoring. The tests are available [here](https://github.com/robustToolbox/package/blob/master/development/tests/fixed/run). To run the test battery:

	$ ./run