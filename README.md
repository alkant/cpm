This is a C++11 implementation of the Convex Polytope Machine algorithm as presented in

Kantchelian, A., Tschantz, M. C., Huang, L., Bartlett, P. L., Joseph, A. D., & Tygar, J. D. [Large-Margin Convex Polytope Machine](http://papers.nips.cc/paper/5511-large-margin-convex-polytope-machine).
 
In addition to the command line tool, Python bindings which are fully aware
of numpy arrays and scipy sparse matrices are provided.

## Building the code

The CPM can be invoked via command line or from a python interpreter 
directly. Building these two should be painless.

### Building the command line tool

Running

``` bash
$ make cmdapp
```

will create the `bin/cpm` executable.

### Building the python module

The Python module is built and installed using the distutils tools, which
is already included in the standard library. However, the python module
itself requires numpy (and numpy headers) and scipy, so make sure these 
are installed. To build the extension, run:

``` bash
$ python setup.py build
```

This puts all necessary files in `build/lib.<your_architecture>`. 
From this directory, you should be able to launch a Python interpreter and
successfully import the module. For example:

``` bash
$ cd build/lib.macosx-10.9-x86_64-2.7
$ ls
_cpm.so cpm.py
$ python
Python 2.7.6 (default, Nov 25 2013, 16:54:21) 
[GCC 4.2.1 Compatible Apple LLVM 5.0 (clang-500.2.79)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import cpm
>>> 
```

If you are particularly fond of it, you can install the cpm 
extension module by

``` bash
$ python setup.py install
```

which will make it import available from anywhere. Or you can just copy the
build/lib.* files where needed.

#### Building the python wrapper

Unless you are planning to extend the python module features yourself, 
this is part is irrelevant to you.

You need a working installation of [SWIG](http://www.swig.org/). Running

``` bash
$ make wrapper
```

will create `python_wrapper.cpp` and `cpm.py` in the src directory.

## Usage

### Python module

The python module essentially provides two classes: `CPM` and `Dataset`. 
`CPM` essentially behaves like a scikit learner, with `fit()` and 
`predict()` methods.
`Dataset` takes care of reading libSVM files from disk (fast!) and/or 
translating your dense numpy data or sparse matrices into the memory 
layout CPM likes. Most of the methods come with `__docstring__`s for
easy reference. Here's an example usage:

``` python
>>> import numpy as np, scipy.sparse as sp
>>> import cpm

>>> X = [[0, 1, 0], [1, 0, 0]] # two instances with 3 features
>>> Y = [0, 1] # corresponding labels

>>> Xnumpy = np.array(X, dtype=np.float32) # notice the type spec
>>> Xsparse = sp.csr_matrix(X, dtype=np.float32) # notice the type spec again
>>> Ynumpy = np.array(Y, dtype=np.int32) # everything is 32 bits.

>>> trainset_1 = cpm.Dataset(X, Y) # this works
>>> trainset_2 = cpm.Dataset(Xnumpy, Ynumpy) # this is fine too
>>> trainset_3 = cpm.Dataset(Xsparse, Ynumpy) # and this as well

>>> clf = cpm.CPM(10) # a CPM with 10 sub-classifiers and default meta-parameters values
>>> clf.fit(trainset_1, 100) # train model on 100 SGD steps
>>> scores, assignments = clf.predict(trainset_2) # returns predicted scores and assigned sub-classifiers
>>> scores
array([-1.11855221,  1.37075484], dtype=float32)
>>> assignments
array([0, 0], dtype=int32)

>>> clf.serializeModel('my_model.cpm') # you can save the current model to disk
>>> clf_1 = cpm.CPM.deserializeModel('my_model.cpm') # and load it again later
```

The main gotcha is that data arrays should be float 32 bits and label arrays
32 bit integers (or less). You can accomplish this easily by the `view()` 
method of numpy/scipy.sparse objects. It is anyhow a good idea to be
using 32 bit floats by default if working with large datasets. At the moment,
only CSR sparse matrices are supported.

The wrapper also exposes `parallelFitPredict()`, a multithreaded method 
which trains multiple models with arbitrary parameters and outputs their 
predictions. Refer to the docstring for how to use it.

### Command line

The command line interface follows [Sofia-ml](https://code.google.com/p/sofia-ml) 
in spirit.

``` bash
$ ./cpm -h
Perform CPM training and/or inference.

--quiet -q   be quiet.
    Default: False
--reshuffle   shuffle training set between epochs.
    Default: False
--classifiers -k <int>   number of classifiers.
    Default: 1
--outer_label <int>   outer class label (the class that will be decomposed).
    Default: 1
--iterations -i <int>   number of iterations.
    Default: 50000000
--C -C <float>   C regularization factor.
    Default: 1
--cost_ratio <float>   cost ratio of negatives vs positives.
    Default: 1
--entropy <float>   minimal (exp of) entropy to maintain in heuristic max. Value between 1 and k.
    Default: 1
--seed <unsigned long>   random seed (for reproducibility).
--train -t <string>   train data file.
--test -c <string>   test data file.
--model_in -m <string>   model in file. Will be ignored if in training mode.
--model_out -o <string>   model out file.
--scores -s <string>   scores file.
```

For instance, to train a model with 10 subclassifiers on 1,000,000 iterations, 
from libsvm formated file `train.libsvm`, save model in `model.txt` and output the scores 
of the model on `test.libsvm` in `scores.txt`:

``` bash
$ ./cpm -k 10 -i 1000000 -t train.libsvm -c test.libsvm -o model.txt -s scores.txt
```

