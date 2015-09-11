# Testing Infrastructure

This directory contains the main testing infrastructure for the ***RobuPy*** package


### Random

The setup allows to test the *RobuPy* for a fixed amount of time. Random model specifications are generated and the integrity of the package is checked. 

For more details:

    $ ./run -h

### Fixed

This setup runs battery of tests for fixed model specifications and allows to ensure that the results of the package does not change during refactoring. To run the test battery:

	$ ./run