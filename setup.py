from setuptools import find_packages, setup
from Cython.Build import cythonize

setup(name="Epidemic threshold",
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
      # test_requires=["unittest"],
      data_files=[('tests/', ['tests/*.csv'])],
      ext_modules=cythonize(["threshold/utilc.pyx"]),
      )
