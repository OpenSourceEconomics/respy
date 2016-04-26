# standard library
from setuptools.command.build_py import build_py
from setuptools import find_packages
from setuptools import setup

import os


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
            'fortran/lib/*.*', 'fortran/include/*.*']},
        version="0.0.7",
        description='respy is a Python package for the solution, simulation, and estimation of the prototypical dynamic discrete choice model.',
        author='Philipp Eisenhauer',
        author_email='eisenhauer@policy-lab.org',
        url='http://respy.readthedocs.org',
        keywords=['Economics', 'Dynamic Discrete Choice Model'],
        classifiers=[],
        setup_requires=['pytest-runner'],
        tests_require=['pytest'],
        install_requires=['numpy', 'scipy', 'pandas', 'scipy',
            'statsmodels', 'pytest'],
        cmdclass={'build_py': CustomBuildCommand},
        include_package_data=True
        )

    setup(**metadata)

if __name__ == '__main__':
    setup_package()
