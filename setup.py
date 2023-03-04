from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize('discfilter.pyx',annotate=True) #Using annotate=True outputs HTML file describing c-code generated
)
