from setuptools import find_packages, setup
from sys import stderr

try:
    # try to import Cython. If it can't, it doesn't
    from Cython.Build import cythonize
    cython_args = {'ext_modules': cythonize(["threshold/utilc.pyx"])}
except ImportError:
    print >> stderr, '****** !!! *** No Cython modules, going for pure Python.'
    cython_args = {}

try:
    # if cython was imported, try to cythonize
    setup(name="Epidemic_Threshold",
          version="1.0",
          description="Compute the epidemic threshold on time-evolving networks",
          author="Eugenio Valdano",
          author_email="eugenio.valdano@gmail.com",
          platforms=["any"],  # or more specific, e.g. "win32", "cygwin", "osx"
          license="BSD",
          url="http://www.linkedin.com/in/eugeniovaldano",
          packages=find_packages(),
          test_suite="tests",
          install_requires=[i.strip() for i in open("requirements.txt").readlines()],
          data_files=[('tests/', ['tests/*.csv'])],
          **cython_args
          )
except:
    # no cythonizing around here
    print >> stderr, '****** !!! *** No C building, going for pure Python.'

    setup(name="Epidemic_Threshold",
          version="1.0",
          description="Compute the epidemic threshold on time-evolving networks",
          author="Eugenio Valdano",
          author_email="eugenio.valdano@gmail.com",
          platforms=["any"],  # or more specific, e.g. "win32", "cygwin", "osx"
          license="BSD",
          url="http://www.linkedin.com/in/eugeniovaldano",
          packages=find_packages(),
          test_suite="tests",
          install_requires=[i.strip() for i in open("requirements.txt").readlines()],
          data_files=[('tests/', ['tests/*.csv'])],
          )

