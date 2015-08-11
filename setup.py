from distutils.core import setup
setup(
  name = 'robupy',
  packages = ['robupy', 'robupy.checks', 'robupy.fort', 'robupy.tests'], # this
  # must be the same as the name above
    package_data = {'robupy': ['fort/*.f90'],},
  version = '0.1.2',
  description = 'Toolbox to explore robust dynamic discrete choice model',
  author = 'Philipp Eisenhauer',
  author_email = 'eisenhauer@policy-lab.org',
  url = 'https://github.com/grmToolbox/package', # use the URL to the github repo
  keywords = ['Economics', 'Dynamic Discrete Choice Model', 'Robustness'], # arbitrary keywords
  classifiers = [],
  install_requires=['numpy', 'pandas', 'coveralls', 'nose', 'scipy'],
)
