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

// parallel_eval.cpp

// Author: Alex Kantchelian, 2014
// akant@cs.berkeley.edu

#include <thread>

#include "parallel_eval.h"
#include "cpm.h"

void ParallelEval::evalf(const StochasticDataAdaptor& trainset, const StochasticDataAdaptor& testset,
                         const CPMConfig config, float* scores, int* assignments) {
    
    // generate random seed
    size_t seed = std::chrono::system_clock::now().time_since_epoch().count();
    seed = seed ^ std::hash<std::thread::id>()(std::this_thread::get_id());
    if (sizeof(seed) == 8) {
        seed = seed ^ (seed >> 32);
    }
    
    CPM model(config.k, config.outer_label, config.lambda, config.entropy, config.cost_ratio, (unsigned int) seed);
    model.fit(trainset, config.iterations, config.reshuffle, false);
    model.predict(testset, scores, assignments);
}

void ParallelEval::parallelEval(const StochasticDataAdaptor& trainset, const StochasticDataAdaptor& testset,
                                const std::vector<CPMConfig> configs, float* out_scores, int* out_assignments) {
    std::thread* threads = new std::thread[configs.size()];
    
    for(size_t i=0; i<configs.size(); ++i) {
        threads[i] = std::thread(evalf, std::ref(trainset), std::ref(testset), configs[i],
                                 out_scores + i*testset.getNInstances(), out_assignments + i*testset.getNInstances());
    }
    
    for (size_t i=0; i<configs.size(); ++i) {
        threads[i].join();
    }
    
    delete[] threads;
}
