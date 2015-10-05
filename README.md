# *RobuPy*

Toolbox for the simulation and estimation of robust dynamic discrete choice models. Building on the illustrative model from:


> Michael P. Keane, Kenneth I. Wolpin (1994). [The Solution and Estimation of Discrete Choice Dynamic Programming Models by Simulation and Interpolation: Monte Carlo Evidence](http://www.jstor.org/stable/2109768). *The Review of Economics and Statistics*, 76(4): 648-672.

Additional information about our research is available [online](http://www.policy-lab.org) and you can
sign up for our mailing list [here](http://eepurl.com/RStEH). Please feel free to contact us directly: 

[![Join the chat at https://gitter.im/robustToolbox/contact](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/robustToolbox/contact?utm_source=share-link&utm_medium=link&utm_campaign=share-link)

#### Installation
 
    $ pip install robupy
    $ python -c "import robupy; robupy.test()"

The last command runs several tests to check the integrity of the package. You can look at the tests [here](https://github.com/robustToolbox/package/blob/master/robupy/tests/tests.py). We have set up an additional testing infrastructure [here](https://github.com/robustToolbox/package/tree/master/development/tests). To get started using the package, we provide an *IPython Notebook* [here](http://nbviewer.ipython.org/github/robustToolbox/package/blob/master/example/example.ipynb).

#### Quality Assurance

[![Build Status](https://travis-ci.org/robustToolbox/package.svg?branch=master)](https://travis-ci.org/robustToolbox/package)
[![Coverage Status](https://coveralls.io/repos/peisenha/robustToolbox/badge.svg?branch=development&service=github)](https://coveralls.io/github/peisenha/robustToolbox?branch=development)
[![Code Issues](http://www.quantifiedcode.com/api/v1/project/409a24d80b7145988ec12a35e94315bb/badge.svg)](http://www.quantifiedcode.com/app/project/409a24d80b7145988ec12a35e94315bb)
[![Codacy Badge](https://www.codacy.com/project/badge/19e3f4eeb62e435ebd3f078fcae89cdb)](https://www.codacy.com/app/eisenhauer/robustToolbox_package)

Instructions to reproduce the simulated samples from the Monte Carlo analysis in Keane & Wolpin (1994) are available [here](http://nbviewer.ipython.org/github/robustToolbox/package/blob/master/development/analyses/restud/lecture.ipynb). We also provide some visual illustrations of the economics underlying their simulations. 
