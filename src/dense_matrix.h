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

// dense_matrix.h

// Author: Alex Kantchelian, 2014
// akant@cs.berkeley.edu

#ifndef __cpm__dense_matrix__
#define __cpm__dense_matrix__

#include <iostream>
#include <cstring>
#include <limits>
#include <cmath>

#include "sparse_vector.h"

class DenseMatrix {
public:
    DenseMatrix(int dimensions, int classifiers);
    
    DenseMatrix(const DenseMatrix& other) : dimensions(other.dimensions), classifiers(other.classifiers) {
        
        data = new float[dimensions * ((size_t) classifiers)];
        std::memcpy(data, other.data,
                    sizeof(float) * ((size_t) dimensions) * ((size_t) classifiers));
        
        scales = new double[classifiers];
        std::memcpy(scales, other.scales, sizeof(double) * classifiers);
        
        intercept = new double[classifiers];
        std::memcpy(intercept, other.intercept, sizeof(double) * classifiers);
    }
    
    DenseMatrix(DenseMatrix&& other) : dimensions(other.dimensions), classifiers(other.classifiers), data(other.data), scales(other.scales), intercept(other.intercept) {
        
        other.data = nullptr;
        other.scales = nullptr;
        other.intercept = nullptr;
        //other.norms2 = nullptr;
    }
    
    ~DenseMatrix() {delete[] data; delete[] scales; delete[] intercept;};
    
    // res will be zeroed-out
    // res must have 'classifiers' size
    void inner(const SparseVector& s, double* res, const bool* fmask=nullptr) const;
    
    double l2norm() const;
    
    // for all k, w_k += a_k * s
    // with optional support for dropout noise
    void addInplace(const SparseVector& s, const double * const a, const bool* fmask=nullptr);
    
    // w_k += a * s
    // with optional support for dropout noise
    void addInplace(const SparseVector& s, double a, int k, const bool* fmask=nullptr);
    
    // for all k, w_k *= a_k
    void mulInplace(const double * const a);
    
    // w *= a
    void mulInplace(double a);
    
    // zeros-out matrix
    void clear();
    
    void serialize(std::ofstream* outstream) const;
    void deserialize(std::ifstream* instream);
    
    const int dimensions;
    const int classifiers;
    
    const double bias = 1.0;
    
private:
    // unscaled data
    float* data;
    
    // data scales
    double* scales;
    
    // bias terms are scaled
    double* intercept;
    
    void rescale();
    const double min_scale = std::sqrt(std::numeric_limits<float>::min());
};


#endif /* defined(__cpm__dense_matrix__) */
