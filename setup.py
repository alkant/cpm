#!/usr/bin/env python

"""
setup.py file for SWIG interface
"""

from distutils.core import setup, Extension
import numpy as np

cpm_module = Extension('_cpm',
                           sources=['src/python_wrap.cpp',
                                   'src/sparse_vector.cpp',
                                   'src/stochastic_data_adaptor.cpp',
                                   'src/convex_polytope_machine.cpp',
                                   'src/dense_matrix.cpp',
                                   'src/cpm.cpp',
                                   'src/eval_utils.cpp',
                                   'src/parallel_eval.cpp'],
                           language='c++',
                           swig_opts=['-c++', '-O', '-builtin'],
                           extra_compile_args=['-std=c++11', '-pthread'],
                           include_dirs=[np.get_include(), 'src']
                           )

setup (name = 'cpm',
       version = '0.2',
       author      = "Alex Kantchelian",
       description = """Convex Polytope Machine""",
       ext_modules = [cpm_module],
       py_modules = ["cpm"],
       package_dir={'': 'src'}
      )
