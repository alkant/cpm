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

// eval_utils.h

// Author: Alex Kantchelian, 2014
// akant@cs.berkeley.edu

#ifndef __cpm__eval_utils__
#define __cpm__eval_utils__

#include <iostream>
#include <map>
#include <memory>

#include "sparse_vector.h"
#include "convex_polytope_machine.h"
#include "stochastic_data_adaptor.h"

namespace evalutils {

enum Metric { Accuracy, AbsoluteTop, AUC, AUC01, AUC001,
            Cost, CostPositives, CostNegatives, Redundancy, Entropy, L2,
    TruePositiveRate, FalsePositiveRate, Precision};

double entropy(const int* assignments, size_t length, unsigned short k);

std::unique_ptr<std::map<Metric, double>> measure(const StochasticDataAdaptor& testset, ConvexPolytopeMachine& model);

}

#endif /* defined(__cpm__eval_utils__) */
