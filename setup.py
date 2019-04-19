from setuptools.command.build_py import build_py
from setuptools.command.develop import develop
from setuptools import find_packages
from setuptools import setup
import os
import subprocess


class CustomDevelopCommand(develop):
    """ Customized setuptools install command - prints a friendly greeting.
    """

    def run(self):
        """ Overwriting the existing command.
        """
        os.chdir("respy")

        subprocess.run(["python", "waf", "distclean"])
        subprocess.run(["python", "waf", "configure", "build", "-j", "1", "-vvv"])

        os.chdir("../")

        develop.run(self)


class CustomBuildCommand(build_py):
    """ Customized setuptools install command - prints a friendly greeting.
    """

    def run(self):
        """ Overwriting the existing command.
        """
        os.chdir("respy")

        subprocess.run(["python", "waf", "distclean"])
        subprocess.run(["python", "waf", "configure", "build", "-j", "1", "-vvv"])

        os.chdir("../")

        build_py.run(self)


def setup_package():
    """ First steps towards a reliable build process.
    """
    metadata = dict(
        name="respy",
        packages=find_packages(),
        package_data={
            "respy": [
                "fortran/bin/*",
                "fortran/*.so",
                "fortran/lib/*.*",
                "fortran/include/*.*",
                "tests/resources/*",
                ".config",
                "pre_processing/base_spec.csv",
            ]
        },
        version="2.0.0.dev20",
        description=(
            "respy is a Python package for the simulation and estimation of a "
            "prototypical finite-horizon dynamic discrete choice model."
        ),
        author="Philipp Eisenhauer",
        author_email="eisenhauer@policy-lab.org",
        url="http://respy.readthedocs.io",
        keywords=["Economics", " Dynamic Discrete Choice Model"],
        classifiers=[],
        install_requires=[
            "numpy>=1.11",
            "scipy>=0.18",
            "pandas>=0.18",
            "statsmodels>=0.6",
            "pytest>=3.0",
            "pyaml",
        ],
        cmdclass={"build_py": CustomBuildCommand, "develop": CustomDevelopCommand},
        include_package_data=True,
    )

    setup(**metadata)


if __name__ == "__main__":
    setup_package()
