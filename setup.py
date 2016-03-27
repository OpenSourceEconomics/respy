from setuptools import setup, find_packages
import os

def setup_package():
    

    os.chdir('robupy')

    os.system('./waf distclean; ./waf configure build')

    os.chdir("../")

    metadata = dict(

   name='robupy',
    packages=find_packages(),
    package_data={'robupy': ['fortran/bin/robufort',
        'fortran/include/*.mod', 'fortran/lib/*.a'], },
    version="0.1.8",
    description='Toolbox to explore robust dynamic discrete choice models',
    author='Philipp Eisenhauer',
    author_email='eisenhauer@policy-lab.org',
    url='https://github.com/robustToolbox/package',
    keywords=['Economics', 'Dynamic Discrete Choice Model', 'Robustness'],
    classifiers=[],
   setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    
    install_requires=['numpy', 'scipy', 'pandas', 'scipy', 'statsmodels', 'pytest'],
    
    )

    setup(**metadata)

if __name__ == '__main__':
    setup_package()