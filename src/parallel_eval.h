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

// parallel_eval.h

// Author: Alex Kantchelian, 2014
// akant@cs.berkeley.edu

#ifndef __cpm__parallel_eval__
#define __cpm__parallel_eval__

#include <vector>

#include "stochastic_data_adaptor.h"

struct CPMConfig {
    
    CPMConfig() {}; // to make SWIG templating happy
    
    CPMConfig(int outer_label, int k, float lambda, float entropy, float cost_ratio,
              int iterations, bool reshuffle) : outer_label(outer_label),
    k(k), lambda(lambda), entropy(entropy), cost_ratio(cost_ratio), iterations(iterations),
    reshuffle(reshuffle) {}
    
    int outer_label;
    int k;
    float lambda;
    float entropy;
    float cost_ratio;
    int iterations;
    bool reshuffle;
};

namespace ParallelEval {
    void evalf(const StochasticDataAdaptor& trainset, const StochasticDataAdaptor& testset,
               const CPMConfig config, float* scores, int* assignments);
    
    void parallelEval(const StochasticDataAdaptor& trainset, const StochasticDataAdaptor& testset,
                      const std::vector<CPMConfig> configs, float* out_scores, int* out_assignments) ;
}

#endif /* defined(__cpm__parallel_eval__) */
