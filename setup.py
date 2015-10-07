from distutils.core import setup
setup(
  name = 'robupy',
  packages = ['robupy', 'robupy.python', 'robupy.python.py', 'robupy.python.f2py', 'robupy.fortran'], # this
  package_data = {'robupy': ['fortran/*.f90', 'fortran/*.f', 'python/f2py/*.f90'],},
  version = "0.1.6" ,
  description = 'Toolbox to explore robust dynamic discrete choice models',
  author = 'Philipp Eisenhauer',
  author_email = 'eisenhauer@policy-lab.org',
  url = 'https://github.com/robustToolbox/package', # use the URL to the github repo
  keywords = ['Economics', 'Dynamic Discrete Choice Model', 'Robustness'], # arbitrary keywords
  classifiers = [],
  install_requires=['numpy', 'pandas', 'coveralls', 'nose', 'scipy'],
)
