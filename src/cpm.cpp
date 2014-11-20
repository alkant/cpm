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

// cpm.cpp

// Author: Alex Kantchelian, 2014
// akant@cs.berkeley.edu

#include <cmath>
#include <algorithm>
#include <stdexcept>

#include "eval_utils.h"
#include "cpm.h"

CPM::CPM(int k, int outer_label, float lambda, float entropy, float cost_ratio, unsigned int seed) : outer_label(outer_label), k(k), lambda(lambda), entropy(entropy), cost_ratio(cost_ratio), seed(seed), generator(seed) {
}

void CPM::fit(const StochasticDataAdaptor& trainset, int iterations, bool reshuffle, bool verbose){
    size_t dim = trainset.getDimensions();
    
    size_t n_instances = trainset.getNInstances();
    
    if (n_instances < 1) {
        std::cerr << "Empty training set" << std::endl;
        return;
    }
    
    size_t n_positives = trainset.getCountsPerClass().find(outer_label)->second;
    size_t n_negatives = n_instances - n_positives;
    
    if (verbose){
        std::cout << "Number of dimensions: " << dim <<'\n'
        << "Number of classifiers: " << k << '\n'
        << "Lambda: " << lambda << '\n'
        << "Iterations: " << iterations << '\n'
        << "Cost ratio: " << cost_ratio << '\n'
        << "Minimum entropy: " << std::exp(entropy) << "\n\n";
        
        std::cout << "negatives: " << n_negatives << " ("
        << 100*((float) n_negatives)/n_instances
        << "%), positives: " << n_positives << " ("
        << 100*((float) n_positives)/n_instances << "%)\n\n";
    }
    
    if(model) {
        delete model;
    }
    
    model = new ConvexPolytopeMachine(outer_label, (int) dim, (unsigned short) k, lambda/iterations, entropy, cost_ratio/(1.0f + cost_ratio), 1.0f/(1.0f+cost_ratio), n_positives, seed);
    
    int seen_positives = 0; // number of positive instances seen
    int seen_negatives = 0; // number of negative instances seen
    double pos_loss = 0; // loss on positive samples
    double neg_loss = 0; // loss on negative samples
    double redundancy = 0; // exclusion loss
    size_t reassignments = 0;
    int epoch = 0;
    
    if (verbose){
        std::cout << "Round\tReassignments\tRedundancy\tEntropy\tNegative loss\tPositive loss\n";
    }
    
    size_t* perm = new size_t[n_instances];
    for (size_t i=0; i < n_instances; i++) { //FIXME
        perm[i] = i;
    }
    std::shuffle(perm, perm + n_instances, generator);
    
    for(int iter = 0; iter < iterations; ++iter) {
        // sample next instance
        const std::tuple<int, const SparseVector, size_t>& lic = trainset.getInstance(perm[iter%n_instances]);
        
        int previous_assignment = (std::get<0>(lic) == outer_label) ? model->getAssignment(std::get<2>(lic)) : -1;
        
        auto score_eloss_assignment_active = model->oneStep(lic);
        
        float score = std::get<0>(score_eloss_assignment_active);
        float redun = std::get<1>(score_eloss_assignment_active);
        unsigned short assignment = std::get<2>(score_eloss_assignment_active);
        
        if (std::get<0>(lic) == outer_label) {
            pos_loss += std::max(0.0, 1.0 - score);
            redundancy += redun;
            
            if (previous_assignment != assignment) {
                reassignments++;
            }
            
            seen_positives++;
            
        } else {
            neg_loss += std::max(0.0, 1.0 + score);
            seen_negatives++;
        }
        
        if ((seen_negatives >= (int) n_negatives) &&
            (seen_positives >= (int) n_positives)) {
            float rate = ((float) reassignments) / n_positives;
            float entropy = (float) evalutils::entropy(model->getAssignments(), n_positives, (unsigned short) k);
            
            if(verbose) {
                std::cout << epoch << '\t'
                << rate << '\t'
                << redundancy/n_positives << '\t'
                << entropy << '\t'
                << neg_loss/seen_negatives << '\t'
                << pos_loss/n_positives << std::endl;
            }
            
            seen_positives = 0;
            reassignments = 0;
            seen_negatives = 0;
            neg_loss = 0.0;
            pos_loss = 0.0;
            redundancy = 0.0;
            epoch++;
            
            if (reshuffle) std::shuffle(perm, perm + n_instances, generator);
        }
    }
    
    delete[] perm;
}

void CPM::predict(const StochasticDataAdaptor& testset, float* scores, int* assignments) const {
    size_t n_instances = testset.getNInstances();
    
    if (!model) {
        throw std::runtime_error("Empty model.");
    }
    
    for (size_t i = 0; i < n_instances; ++i) {
        auto sa = model->predict(std::get<1>(testset.getInstance(i)));
        scores[i] = (float) sa.first;
        assignments[i] = sa.second;
    }
}

std::pair<double, int> CPM::predict(const SparseVector& sv) const {
    return model->predict(sv);
}

void CPM::serializeModel(const char* filename) const {
    if(model) {
        model->serializeModel(filename);
    }
}

CPM* CPM::deserializeModel(const char* filename) {
    ConvexPolytopeMachine* model = ConvexPolytopeMachine::deserializeModel(filename);
    
    CPM* res = new CPM(model->k, model->outer_label, model->lambda, model->entropy,
                       model->positive_cost/(model->positive_cost + model->negative_cost),
                       model->seed);
    res->model = model;
    
    return res;
}
