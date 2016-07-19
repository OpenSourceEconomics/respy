from setuptools.command.build_py import build_py
from setuptools.command.develop import develop
from setuptools import find_packages
from setuptools import setup
import os


class CustomDevelopCommand(develop):
    """ Customized setuptools install command - prints a friendly greeting.
    """
    def run(self):
        """ Overwriting the existing command.
        """
        os.chdir('respy')

        os.system('./waf distclean; ./waf configure build')

        os.chdir('../')

        develop.run(self)


class CustomBuildCommand(build_py):
    """ Customized setuptools install command - prints a friendly greeting.
    """
    def run(self):
        """ Overwriting the existing command.
        """
        os.chdir('respy')

        os.system('./waf distclean; ./waf configure build')

        os.chdir('../')

        build_py.run(self)


def setup_package():
    """ First steps towards a reliable build process.
    """
    metadata = dict(
        name='respy',
        packages=find_packages(),
        package_data={'respy': ['fortran/bin/*', 'fortran/*.so',
            'fortran/lib/*.*', 'fortran/include/*.*', 'tests/resources/*']},
        version="0.1.00",
        description='respy is a Python package for the solution, simulation, and estimation of the prototypical dynamic discrete choice model.',
        author='Philipp Eisenhauer',
        author_email='eisenhauer@policy-lab.org',
        url='http://respy.readthedocs.io',
        keywords=['Economics', ' Dynamic Discrete Choice Model'],
        classifiers=[],
        setup_requires=['pytest-runner>=2.7'],
        tests_require=['pytest>=2.9'],
        install_requires=['numpy>=1.11', 'scipy>=0.17', 'pandas>=0.18',
            'statsmodels>=0.6', 'pip>=8.0'],
        cmdclass={'build_py': CustomBuildCommand, 'develop':
            CustomDevelopCommand},
        include_package_data=True
        )

    setup(**metadata)

if __name__ == '__main__':
    setup_package()
