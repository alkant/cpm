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

// cpm.h

// Author: Alex Kantchelian, 2014
// akant@cs.berkeley.edu

#ifndef __cpm__cpm__
#define __cpm__cpm__

#include <iostream>
#include <random>
#include <utility>

#include "stochastic_data_adaptor.h"
#include "convex_polytope_machine.h"
#include "sparse_vector.h"

class CPM {
public:
    CPM(int k, int outer_label, float lambda, float entropy, float cost_ratio, unsigned int seed);
    // lambda is taken as the global penalty constraint in the optimization problem (in praticular,
    // it will later be divided by the number of iterations)
    ~CPM() {delete model;};
    
    void fit(const StochasticDataAdaptor& trainset, int iterations, bool reshuffle, bool verbose);
    void predict(const StochasticDataAdaptor& testset, float* scores, int* assignments) const;
    std::pair<double, int> predict(const SparseVector& sv) const;
    void serializeModel(const char* filename) const;
    static CPM* deserializeModel(const char* filename);
    
    const int outer_label;
    const int k;
    const float lambda;
    const float entropy;
    const float cost_ratio;
    const unsigned int seed;
    
private:
    std::mt19937 generator;
    ConvexPolytopeMachine* model = nullptr;
};

#endif /* defined(__cpm__cpm__) */
