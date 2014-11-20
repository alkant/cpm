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

// dense_matrix.cpp

// Author: Alex Kantchelian, 2014
// akant@cs.berkeley.edu

#include <random>
#include <cmath>
#include <fstream>

#include "dense_matrix.h"

DenseMatrix::DenseMatrix(int dimensions, int classifiers) : dimensions(dimensions), classifiers(classifiers) {
    
    data = new float[((size_t) dimensions) * ((size_t) classifiers)]();
    
    scales = new double[classifiers];
    for (int k = 0; k < classifiers; ++k) {
        scales[k] = 1.0;
    }
    
    intercept = new double[classifiers]();
}

void DenseMatrix::clear() {
    for(size_t i = 0; i < ((size_t) dimensions) * ((size_t) classifiers); ++i) {
        data[i] = 0.0f;
    }
    
    for (int k = 0; k < classifiers; ++k) {
        scales[k] = 0.0;
        intercept[k] = 0;
    }
}

void DenseMatrix::inner(const SparseVector& s, double* res, const bool* fmask) const {
    for(int k = 0; k < classifiers; ++k) {
        res[k] = 0.0;
    }
    
    int i = 0;
    for(auto const& iv: s.data){
        if (iv.index >= dimensions) continue; // ignore extra dimensions
        if (fmask && fmask[i]) continue; // dropout feature
        
        size_t offset = ((size_t) iv.index) * ((size_t) classifiers);
        double value = (double) iv.value;
        
        for(size_t k = 0; k < (size_t) classifiers; ++k){
            res[k] += value * ((double) data[offset + k]);
        }
        ++i;
    }
    
    for (int k = 0; k < classifiers; ++k) {
        res[k] = res[k]*scales[k] + intercept[k];
    }
}

void DenseMatrix::rescale() {
    for (size_t i = 0; i < ((size_t) dimensions) * ((size_t) classifiers); ++i) {
        data[i] = (float) (((double) data[i]) +  scales[i%classifiers]);
    }
    
    for (int k = 0; k < classifiers; ++k) {
        scales[k] = 1.0;
    }
}

void DenseMatrix::mulInplace(const double * const a) {
    bool torescale = false;
    for (int k = 0; k < classifiers; ++k) {
        scales[k] *= a[k];
        intercept[k] *= a[k];
        if (scales[k] < min_scale) torescale = true;
    }
    
    if (torescale) rescale();
}

void DenseMatrix::mulInplace(double a) {
    bool torescale = false;
    for (int k = 0; k < classifiers; ++k) {
        scales[k] *= a;
        intercept[k] *= a;
        if (scales[k] < min_scale) torescale = true;
    }
    
    if (torescale) rescale();
}

void DenseMatrix::addInplace(const SparseVector& s, const double* const a, const bool* fmask) {
    int i = 0;
    for(auto const& iv: s.data) {
        if(fmask && fmask[i]) continue;
        
        double value = iv.value;
        size_t offset = ((size_t) iv.index) * ((size_t) classifiers);
        
        for(size_t k = 0; k < ((size_t) classifiers); ++k){
            data[k + offset] = (float) (((double) data[k + offset]) + (value * a[k])/scales[k]);
        }
        ++i;
    }
    
    for(int k = 0; k < classifiers; ++k) {
        intercept[k] += bias * a[k];
    }
}

void DenseMatrix::addInplace(const SparseVector& s, double a, int k, const bool* fmask) {
    int i = 0;
    for(auto const& iv: s.data) {
        if(fmask && fmask[i]) continue;
        
        double value = iv.value;
        size_t index = ((size_t) iv.index) * ((size_t) classifiers) + ((size_t) k);
        
        data[index] = (float) (((double) data[index]) + (a * value)/scales[k]);
        ++i;
    }
    
    intercept[k] += bias * a;
}

double DenseMatrix::l2norm() const {
    double res = 0;
    for(size_t i = 0; i< ((size_t) dimensions) * ((size_t) classifiers); ++i){
        res += (((double) data[i]) * scales[i%classifiers]) * (((double) data[i]) * scales[i%classifiers]);
    }
    
    return std::sqrt(res);
}

void DenseMatrix::serialize(std::ofstream* outstream) const {
    for(size_t i = 0; i < ((size_t) dimensions) * ((size_t) classifiers); ++i) {
        int k = i%classifiers;
        *outstream << scales[k] * data[i] << ' ';
    }
    
    for(int i = 0; i < classifiers; ++i){
        *outstream << intercept[i] << ' ';
    }
    
    *outstream << '\n';
}

void DenseMatrix::deserialize(std::ifstream* instream) {
    for (size_t i = 0; i < ((size_t) dimensions) * ((size_t) classifiers); ++i){
        *instream >> data[i];
    }
    
    for (int i = 0; i < classifiers; ++i) {
        *instream >> intercept[i];
    }
}
