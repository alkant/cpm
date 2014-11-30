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

// stochastic_data_adaptor.cpp

// Author: Alex Kantchelian, 2014
// akant@cs.berkeley.edu

#include <string.h>
#include <utility>
#include <algorithm>
#include <fstream>
#include <stdexcept>

#include "stochastic_data_adaptor.h"

StochasticDataAdaptor::StochasticDataAdaptor(const char* fname, size_t n_instances) {
    instances.clear();
    instances.reserve(n_instances);
    countsPerClass.clear();
    dimensions = 0;
    
    // load data in memory
    const size_t buffer_size = 8 * 1024 * 1024;
    std::ifstream fin(fname);
    char* localBuffer = new char[buffer_size];
    fin.rdbuf()->pubsetbuf(localBuffer, buffer_size);
    
    std::string line;
    
    while(getline(fin, line) && line.size() > 4){
        const char* cline = line.c_str();
        
        int label = atoi(cline);
        cline = strchr(cline, ' ');
        if (!cline) {
            throw std::runtime_error("Invalid format: expected ' '");
        }
        
        size_t cid;
        auto it = countsPerClass.find(label);
        if (it == countsPerClass.end()) {
            cid = 0;
            countsPerClass[label] = 1;
        } else {
            cid = it->second;
            it->second++;
        }
        
        SparseVector sv(cline);
        
        instances.emplace_back(label, sv, cid);
        dimensions = std::max(sv.getMaxDimension(), dimensions);
    }
    
    ++dimensions;
    instances.shrink_to_fit();
    
    fin.close();
    delete[] localBuffer;
}

StochasticDataAdaptor::StochasticDataAdaptor(float* data, int* labels, size_t n_instances, size_t n_dimensions) {
    instances.clear();
    instances.reserve(n_instances);
    countsPerClass.clear();
    dimensions = n_dimensions;
    
    for(size_t i = 0; i < (size_t) n_instances; ++i) {
        int label = labels[i];
        
        size_t cid;
        
        auto it = countsPerClass.find(label);
        if (it == countsPerClass.end()) {
            cid = 0;
            countsPerClass[label] = 1;
        } else {
            cid = it->second;
            it->second++;
        }
        
        SparseVector sv(data + i*n_dimensions, n_dimensions);
        instances.emplace_back(label, sv, cid);
    }
}

StochasticDataAdaptor::StochasticDataAdaptor(float* data, int* indices, int* indptr, int* labels, size_t data_len, size_t indptr_len) {
    instances.clear();
    instances.reserve(indptr_len - 1);
    countsPerClass.clear();
    dimensions = 0;
    
    for(size_t i = 0; i < indptr_len - 1; ++i) {
        int label = labels[i];
        
        size_t cid;
        auto it = countsPerClass.find(label);
        if (it == countsPerClass.end()) {
            cid = 0;
            countsPerClass[label] = 1;
        } else {
            cid = it->second;
            it->second++;
        }
        
        SparseVector sv(indices + indptr[i], data + indptr[i], indptr[i+1] - indptr[i]);
        instances.emplace_back(label, sv, cid);
        dimensions = std::max(sv.getMaxDimension(), dimensions);
    }
    
    ++dimensions;
}

void StochasticDataAdaptor::getLabels(int* labels) const {
    for (size_t i = 0; i < instances.size(); ++i){
        labels[i] = std::get<0>(instances[i]);
    }
}
