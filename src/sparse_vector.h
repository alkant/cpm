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

// sparse_vector.h

// Author: Alex Kantchelian, 2014
// akant@cs.berkeley.edu

#ifndef __cpm__sparse_vector__
#define __cpm__sparse_vector__

#include <vector>
#include <memory>

// element cell
struct IValue{
    IValue(int i, float v) : index(i), value(v) {};
    
    int index;
    float value;
};

class SparseVector {

friend class DenseMatrix;

public:
    /* constructor from a libsvm-like string (without label)
     * non_zeros is a performance hint and represents the 
     * initial size of the internal data vector.
    */
    SparseVector(const char* lsf_string, int non_zeros=1000);
    
    // constructor from dense data
    SparseVector(float* data, size_t len);
    
    // constructor from sparse data
    SparseVector(int* indices, float* data, size_t len);
    
    // get number of non-zeros
    inline size_t getSize() const {return data.size();}
    
    // serialize to libsvm-like string
    std::unique_ptr<std::string> toLibSVMFormat() const;
    
    // largest non-zero dimension index, 0 if empty vector
    size_t getMaxDimension() const;
    
    // x = weight * x
    void multiplyInplace(float weight);
    
    // get ||x||_2
    inline double getNorm() const {return norm;}
    
private:
    // internal array of data
    std::vector<IValue> data;
    
    // ||x||_2
    double norm;
};

#endif /* defined(__cpm__sparse_vector__) */
