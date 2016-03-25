from distutils.core import setup
import os

def setup_package():
    

    os.chdir('robupy')

    os.system('./waf distclean; ./waf configure build')

    os.chdir("../")

    metadata = dict(

   name='robupy',
  packages=['robupy', 'robupy.estimate', 'robupy.evaluate', 'robupy.fortran',
                'robupy.process', 'robupy.read', 'robupy.shared',
         'robupy.simulate',
  'robupy.solve', 'robupy.tests', 'robupy.tests.codes',
         'robupy.tests.resources', 'robupy.fortran.bin',
      'robupy.fortran.include', 'robupy.fortran.lib'],
    package_data={'robupy': ['fortran/bin/robufort',
        'fortran/include/*.mod', 'fortran/lib/*.a'], },
    version="0.1.8",
    description='Toolbox to explore robust dynamic discrete choice models',
    author='Philipp Eisenhauer',
    author_email='eisenhauer@policy-lab.org',
    url='https://github.com/robustToolbox/package',
    keywords=['Economics', 'Dynamic Discrete Choice Model', 'Robustness'],
    classifiers=[],

  )

    setup(**metadata)

    print("I am printing ", os.getcwd())

if __name__ == '__main__':
    setup_package()