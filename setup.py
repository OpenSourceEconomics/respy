from pathlib import Path

from setuptools import find_packages
from setuptools import setup


DESCRIPTION = (
    "respy is a Python package for the simulation and estimation of a prototypical "
    "finite-horizon dynamic discrete choice model."
)
README = Path("README.rst").read_text()
PROJECT_URLS = {
    "Bug Tracker": "https://github.com/OpenSourceEconomics/respy/issues",
    "Documentation": "https://respy.readthedocs.io/en/latest",
    "Source Code": "https://github.com/OpenSourceEconomics/respy",
}


setup(
    name="respy",
    version="2.0.0-dev",
    description=DESCRIPTION,
    long_description=DESCRIPTION + "\n\n" + README,
    long_description_content_type="text/x-rst",
    author="Philipp Eisenhauer",
    author_email="eisenhauer@policy-lab.org",
    python_requires=">=3.6.0",
    url="https://respy.readthedocs.io/en/latest/",
    project_urls=PROJECT_URLS,
    packages=find_packages(),
    license="MIT",
    keywords=["Economics", " Dynamic Discrete Choice Model"],
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    install_requires=["numba>=0.43", "pandas>=0.24", "pytest>=4.0", "pyaml"],
    platforms="any",
    include_package_data=True,
    zip_safe=False,
)
