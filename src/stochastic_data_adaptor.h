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

// stochastic_data_adaptor.h

// Author: Alex Kantchelian, 2014
// akant@cs.berkeley.edu

#ifndef __cpm__stochastic_data_adaptor__
#define __cpm__stochastic_data_adaptor__

#include <iostream>
#include <vector>
#include <map>
#include <random>

#include "sparse_vector.h"

class StochasticDataAdaptor {
public:
    /* constructs dataset from a libsvm formatted text file.
     * n_instances is only a performance hint.
     */
    StochasticDataAdaptor(const char* fname, size_t n_instances=1000000);
    
    // constructs dataset from dense in memory data
    StochasticDataAdaptor(float* data, int* labels, size_t n_instances, size_t n_dimensions);
    
    // constructs dataset from sparse in memory data
    StochasticDataAdaptor(float* data, int* indices, int* indptr, int* labels, size_t data_len, size_t indptr_len);
    
    // get a given instance: label, sparsevector, class id
    inline const std::tuple<int, SparseVector, size_t>& getInstance(size_t i) const {
        return instances[i];
    }
    
    void getLabels(int* labels) const;
    
    size_t getNInstances() const {return instances.size();}
    size_t getDimensions() const {return dimensions;}
    const std::map<int, size_t> getCountsPerClass() const {return countsPerClass;}
    
private:
    // number of dimensions
    size_t dimensions;
    
    // label, sparsevector, class id
    std::vector<std::tuple<int, SparseVector, size_t>> instances;
    
    // number of instances per label
    std::map<int, size_t> countsPerClass;
};

#endif /* defined(__cpm__stochastic_data_adaptor__) */
