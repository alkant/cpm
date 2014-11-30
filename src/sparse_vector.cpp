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

// sparse_vector.cpp

// Author: Alex Kantchelian, 2014
// akant@cs.berkeley.edu

#include "sparse_vector.h"

#include <string.h>
#include <sstream>
#include <stdexcept>
#include <cmath>

SparseVector::SparseVector(const char* lsf_string, int non_zeros) {
    data.clear();
    data.reserve(non_zeros);
    norm = 0.0;
    
    int last_index = -1;
    
    const char* curr = lsf_string;
    size_t str_len = strlen(lsf_string);
    
    while(curr && (curr < lsf_string + str_len)
          && (curr[0] != '#') && (curr[0] != '\n') && (curr[0] != '\r')){
        
        if (curr[0] == ' ') {
            curr += 1;
            continue;
        }
        
        int index = atoi(curr);
        curr = strchr(curr, ':');
        if (curr) {
            curr += 1;
        } else {
            throw std::runtime_error("Invalid format: expected ':'");
        }

        float value = (float) atof(curr);
        
        if (index <= last_index) {
            throw std::runtime_error("Indices must be sorted by increasing order.");
        }
        last_index = index;
        
        data.emplace_back(index, value);
        norm += value * value;
        
        curr = strchr(curr, ' ');
        if (!curr) break;
        curr += 1;
    }
    
    norm = std::sqrt(norm);
    data.shrink_to_fit();
}

SparseVector::SparseVector(float* cdata, size_t len) {
    data.clear();
    data.reserve(len);
    norm = 0.0;
    
    for (size_t i = 0; i < len; ++i) {
        float value = cdata[i];
        if (value != 0.0f) {
            data.emplace_back(i, value);
            norm += value * value;
        }
    }
    
    norm = std::sqrt(norm);
    data.shrink_to_fit();
}

SparseVector::SparseVector(int* indices, float* cdata, size_t len) {
    data.clear();
    data.reserve(len);
    norm = 0.0;
    
    int last_index = -1;
    
    for (size_t i = 0; i < len; ++i) {
        float value = cdata[i];
        int index = indices[i];
        
        if (index <= last_index) {
            throw std::runtime_error("Indices must be sorted by increasing order.");
        }
        last_index = index;
        
        data.emplace_back(index, value);
        norm += value * value;
    }
    
    norm = std::sqrt(norm);
    data.shrink_to_fit();
}

void SparseVector::multiplyInplace(float weight) {
    for(auto& iv : data){
        iv.value *= weight;
    }
    norm *= weight;
}

std::unique_ptr<std::string> SparseVector::toLibSVMFormat() const {
    std::stringstream ss;
    for (auto const& iv: data){
        ss << iv.index << ':' << iv.value << ' ';
    }
    ss << '\n';
    return std::unique_ptr<std::string>(new std::string(ss.str()));
}

size_t SparseVector::getMaxDimension() const {
    if (!data.empty()) {
        return data.back().index;
    }
    return 0;
}
