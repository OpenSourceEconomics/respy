# standard library
from setuptools import find_packages
from setuptools import setup

from setuptools.command.install import install

import os

''' Auxiliary
'''


class CustomInstallCommand(install):
    """ Customized setuptools install command - prints a friendly greeting.
    """
    def run(self):
        pass
        # os.chdir('robupy')
        #
        # os.system('./waf distclean; ./waf configure build')
        #
        # os.chdir('../')
        #
        # install.run(self)

''' Setup
'''


def setup_package():


    os.chdir('robupy')
    os.system('./waf configure build')
    os.chdir('../')

    metadata = dict(
        name='robupy',
        packages=find_packages(),
        package_data={'robupy': ['fortran/bin/robufort',
            'fortran/include/*.mod', 'fortran/lib/*.a', 
            'waf', 'wscript', 'fortran/wscript', 'fortran/*.f90', 
            'fortran/*.f', 'fortran/*.f95']},
        version="0.1.8.6",
        description='Toolbox to explore robust dynamic discrete choice models',
        author='Philipp Eisenhauer',
        author_email='eisenhauer@policy-lab.org',
        url='https://github.com/robustToolbox/robupy',
        keywords=['Economics', 'Dynamic Discrete Choice Model', 'Robustness'],
        classifiers=[],
        setup_requires=['pytest-runner'],
        tests_require=['pytest'],
        install_requires=['numpy', 'scipy', 'pandas', 'scipy',
            'statsmodels', 'pytest'],
        cmdclass={'install': CustomInstallCommand}
        )

    setup(**metadata)

if __name__ == '__main__':
    setup_package()