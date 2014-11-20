/*
 Copyright 2014 Alex Kantchelian
 
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
 
 http://www.apache.org/licenses/LICENSE-2.0
 
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
*/

// python.i

// Author: Alex Kantchelian, 2014
// akant@cs.berkeley.edu


%module(docstring="This module provides a wrapper for the Convex Polytope Machine C++ code.") cpm

%{
#define SWIG_FILE_WITH_INIT

#include <vector>
#include <map>
#include "stochastic_data_adaptor.h"
#include "cpm.h"
#include "parallel_eval.h"
%}

%include "std_vector.i"
%include "std_map.i"
%include "numpy.i"
%init %{
import_array();
%}

%pythonbegin %{
# Copyright 2014 Alex Kantchelian
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Alex Kantchelian, 2014
# akant@cs.berkeley.edu
%}

%pythonbegin %{
import random
import numpy as np
from scipy import sparse
%}

%template() std::map<int, size_t>;

namespace std {
  %template(VectorOfStruct) std::vector<CPMConfig>;
}

/* ########### Numpy typemaps ############ */
%apply (float* IN_ARRAY2, int DIM1, int DIM2) {(float* data, int dim1, int dim2), 
                                               (float* lambda_entropy_cost_ratio, int dif1, int dif2)};

%apply (int* IN_ARRAY2, int DIM1, int DIM2) {(int* k_outer_labels_iterations, int dii1, int dii2)};

%apply (int* IN_ARRAY1, int DIM1) {(int* labels, int dim_labels), 
                                   (int* indices, int dim2), 
                                   (int* indptr, int dim3)};

%apply (float* IN_ARRAY1, int DIM1) {(float* sparse_data, int dim1)};

%apply (float* ARGOUT_ARRAY1, int DIM1) {(float* scores, int scores_dim), 
                                         (float* out_scores, int dof)};
%apply (int* ARGOUT_ARRAY1, int DIM1) {(int* assignments, int assignments_dim), 
                                       (int* out_assignments, int doi),
                                       (int* out_labels, int dol)};

/* ###### basic exception handling ##### */
%exception {
  try {
    $action
  } catch (std::exception &e) {
    PyErr_SetString(PyExc_RuntimeError, const_cast<char*>(e.what()));
  }
}

/* ########################################### */

%rename(_Dataset) StochasticDataAdaptor;

// Directly wrapped calls
class StochasticDataAdaptor {
public:
  StochasticDataAdaptor(const char* fname);

  ~StochasticDataAdaptor();
  
  size_t getNInstances() const;
  
  size_t getDimensions() const;

  const std::map<int, size_t> getCountsPerClass() const;
};

%extend StochasticDataAdaptor {
  StochasticDataAdaptor(float* data, int dim1, int dim2, int* labels, int dim_labels) {
    if (dim1 != dim_labels) {
      PyErr_Format(PyExc_ValueError, "Dimensions mismatch.");
      return nullptr;
    }

    return new StochasticDataAdaptor(data, labels, dim1, dim2);
  }

  StochasticDataAdaptor(float* sparse_data, int dim1, int* indices, int dim2, int* indptr, int dim3, int* labels, int dim_labels) {
  if (dim1 != dim2) {
    PyErr_Format(PyExc_ValueError, "Dimension mismatch for data and indices arrays.");
    return nullptr;
  }

  if (dim3 != dim_labels+1) {
    PyErr_Format(PyExc_ValueError, "Dimension mismatch for indptr and labels arrays.");
    return nullptr;
  }

  return new StochasticDataAdaptor(sparse_data, indices, indptr, labels, dim1, dim3);
  }

  void _getLabels(int* out_labels, int dol) const {
    if ($self->getNInstances() != dol) {
      PyErr_Format(PyExc_ValueError, "Internal error.");
      return;
    }

    $self->getLabels(out_labels);
  }
}

%pythoncode %{
class Dataset(_Dataset):
  def __init__(self, *args):
    """Constructs a labeled dataset object that can be used for CPM training 
    and prediction (labels will be ignored when used for prediction). 
    This always incurs a memory copy (for either dense or sparse matrices) 
    or allocation (when loading from libSVM format file).
    
    Dataset(filename):
      filename: str

      Creates a dataset from a libSVM file format on disk.

    Dataset(X, Y):
      X: 2d float array-like object. Sparse scipy CSR matrices are supported.
      Y: 1d int array-like object
      
      Creates a dataset from instances X (one instance per row) and labels Y.
    """
    if len(args) == 1:
      super(Dataset, self).__init__(*args)

    if len(args) == 2:
      if sparse.isspmatrix_csr(args[0]):
        super(Dataset, self).__init__(args[0].data, args[0].indices, args[0].indptr, args[1])
      else:
        super(Dataset, self).__init__(*args)
    
    if len(args) > 2:
      raise ValueError("Too many arguments.")

  def getLabels(self):
    """Returns a numpy array of labels."""
    return self._getLabels(int(self.getNInstances()))
%}

/* ###################################################### */

%rename(_CPM) CPM;

class CPM {
public:
    CPM(int k, int outer_label, float lambda, float entropy, float cost_ratio, 
        unsigned int seed);
    ~CPM();
    
    void fit(const StochasticDataAdaptor& trainset, int iterations, bool reshuffle, bool verbose);
    void serializeModel(const char* filename) const;
    static CPM* deserializeModel(const char* filename);
    
    const int outer_label;
};

%extend CPM {
  void predict(const StochasticDataAdaptor& testset, float* scores, int scores_dim, 
                int* assignments, int assignments_dim) {
    if ((scores_dim != testset.getNInstances()) || (assignments_dim != testset.getNInstances())) {
      PyErr_Format(PyExc_RuntimeError, "Internal error.");
      return;
    }
    
    return $self->predict(testset, scores, assignments);
  }
}

%pythoncode %{
class CPM(_CPM):
  def __init__(self, k, C=1.0, entropy=0.0, 
              cost_ratio=1.0, outer_label=1, 
              seed=None):
    """Initialize an empty CPM model.
       
       Inputs:
          k: int -- number of sub-classifiers
          C: float -- inverse of L2 regularization factor
          entropy: float -- minimal assignment entropy to maintain
          cost_ratio: float -- in penalty, cost ratio between negative and positive 
            misclassification training errors
          outer_label: int -- outside (positive) class
          seed: (None, int) -- random seed for reproducibility
    """
    if seed is None:
      seed = int(random.getrandbits(32))

    super(CPM, self).__init__(k, outer_label, 1.0/C, entropy, cost_ratio, seed)

  def fit(self, trainset, iterations=-1, reshuffle=True, verbose=False):
    """Trains a model via SGD.
       
       Inputs:
          trainset: Dataset
          iterations: int -- number of SGD steps. If < 0, will be set to 10 * training set size.
          reshuffle: bool -- reshuffle trainingset between each epoch
          verbose: bool -- print training statistics on stdout
    """
    if iterations < 0:
      iterations = 10 * trainset.getNInstances()
    super(CPM, self).fit(trainset, iterations, reshuffle, verbose)

  def predict(self, testset):
    """Performs inference.
       Input:
          testset: Dataset

       Outputs:
          scores: 1d float array of model scores
          assignments: 1d int array of active sub-classifiers per instance
    """
    return super(CPM, self).predict(testset, int(testset.getNInstances()), int(testset.getNInstances()))
%}

/* ######################################### */

%rename(_CPMConfig) CPMConfig;

struct CPMConfig {
  CPMConfig::CPMConfig(int outer_label, int k, float lambda, float entropy, float cost_ratio,
              int iterations, bool reshuffle);
  
  CPMConfig::CPMConfig() {};
};

%inline %{
  void _parallelEval(const StochasticDataAdaptor& trainset, 
                     const StochasticDataAdaptor& testset,
                     const std::vector<CPMConfig> configs,
                     float* out_scores, int dof,
                     int* out_assignments, int doi) {
    
    ParallelEval::parallelEval(trainset, testset, 
                               configs,
                               out_scores,
                               out_assignments);

  }
%}

%pythoncode %{
def parallelFitPredict(trainset, testset, parameters):
  """Trains and tests len(parameters) models on trainset and testset respectively.
  This will launch exactly len(parameters) threads, use at your own risk.

  Inputs:
    trainset: Dataset - the learning dataset
    testset: Dataset - the testing dataset
    parameters: list of dict objects. Each dict object can contain the
      following string keys defining the parameters of the run. See CPM class. 
      Default values are provided for some keys.
        k
        outer_label (1)
        iterations
        C (1)
        entropy (0) 
        cost_ratio (1)
        reshuffle: (True)

  Outputs:
    S: len(parameters) x testset.getCounts() float array of scores
    A: len(parameters) x testset.getCounts() int array of assignments
  """
  configs = []
  for params in parameters:
    configs.append(_CPMConfig(params.get('outer_label', 1), params['k'], 1.0/params.get('C', 1),
                             params.get('entropy', 0), params.get('cost_ratio', 1), 
                             params['iterations'], params.get('reshuffle', True)))
    
  S, A = _parallelEval(trainset, testset, configs, 
                       int(len(parameters)*testset.getNInstances()),
                       int(len(parameters)*testset.getNInstances()))
  S.resize((len(parameters), testset.getNInstances()))
  A.resize((len(parameters), testset.getNInstances()))
  
  return S, A
%}
