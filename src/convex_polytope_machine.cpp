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

// convex_polytope_machine.cpp

// Author: Alex Kantchelian, 2014
// akant@cs.berkeley.edu

#include "convex_polytope_machine.h"
#include <sstream>
#include <fstream>
#include <cmath>
#include <iomanip>

#include <stdexcept>

ConvexPolytopeMachine::ConvexPolytopeMachine(int outer_label, int dim, unsigned short k, float lambda,
                                             float entropy, float negative_cost,
                                             float positive_cost, size_t n_positives,
                                             unsigned int seed):
        outer_label(outer_label), k(k), lambda(lambda), entropy(entropy), negative_cost(negative_cost),
        positive_cost(positive_cost), n_positives(n_positives), seed(seed), W(dim, k) {
    
    iter = 0;
    distinct_p = 0;
    score = new double[k];
    assignments = new int[n_positives];
    occupancy = new unsigned int[k]();
    
    for (size_t i = 0; i < n_positives; ++i) {
        assignments[i] = -1;
    }
}

void ConvexPolytopeMachine::clear() {
    iter = 0;
    distinct_p = 0;
    W.clear();
}

void ConvexPolytopeMachine::serializeModel(const char* filename) const {
    std::ofstream ss(filename);
    
    ss << "version: " << 2 << '\n';
    
    ss << "\n### DATASET ###\n";
    ss << "outer label: " << outer_label << '\n';
    ss << "outer instances: " << n_positives << '\n';
    ss << "dimensions: " << W.dimensions << '\n';
    
    ss << "\n### CPM PARAMETERS ###\n";
    ss << "hyperplanes: " << k << '\n';
    ss << "iterations: " << iter-1 << '\n';
    ss << "lambda: " << lambda << '\n';
    ss << "entropy: " << entropy << '\n';
    ss << "cost ratio: " << negative_cost/positive_cost << '\n';
    ss << "seed: " << seed << '\n';
    
    ss << "\n### ASSIGNMENTS COUNTS ###\n";
    int active = 0;
    
    for (int i=0; i<k; ++i){
        if (occupancy[i] > 0) {
            active++;
        }
    }
    ss << "active classifiers: " << active << '\n';
    ss << "counts: ";
    for (int i=0; i<k; ++i){
        ss << occupancy[i] << ' ';
    }
    ss << '\n';
    
    ss << "\n### MODEL ###\n";
    ss << "encoding: " << "dense\n";
    W.serialize(&ss);
}

ConvexPolytopeMachine* ConvexPolytopeMachine::deserializeModel(const char *filename) {
    std::ifstream ss(filename);
    
    int version;
    ss.ignore(std::numeric_limits<std::streamsize>::max(), ':');
    ss >> version;
    
    if (version != 2) {
        throw std::runtime_error("Unsupported model file version.");
    }
    
    ss.ignore(std::numeric_limits<std::streamsize>::max(), ':');
    int outer_label;
    ss >> outer_label;
    
    ss.ignore(std::numeric_limits<std::streamsize>::max(), ':');
    size_t n_positives;
    ss >> n_positives;
    
    ss.ignore(std::numeric_limits<std::streamsize>::max(), ':');
    int dimensions;
    ss >> dimensions;
    
    ss.ignore(std::numeric_limits<std::streamsize>::max(), ':');
    int k;
    ss >> k;
    
    ss.ignore(std::numeric_limits<std::streamsize>::max(), ':');
    int iter;
    ss >> iter;
    
    ss.ignore(std::numeric_limits<std::streamsize>::max(), ':');
    float lambda;
    ss >> lambda;
    
    ss.ignore(std::numeric_limits<std::streamsize>::max(), ':');
    float entropy;
    ss >> entropy;
    
    ss.ignore(std::numeric_limits<std::streamsize>::max(), ':');
    float cost_ratio;
    ss >> cost_ratio;
    
    ss.ignore(std::numeric_limits<std::streamsize>::max(), ':');
    unsigned int seed;
    ss >> seed;
    
    ss.ignore(std::numeric_limits<std::streamsize>::max(), ':');
    int active;
    ss >> active;
    
    ss.ignore(std::numeric_limits<std::streamsize>::max(), ':');
    ss.ignore(std::numeric_limits<std::streamsize>::max(), ':');
    ss.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    
    ConvexPolytopeMachine* cpm = new ConvexPolytopeMachine(outer_label, dimensions, (unsigned short) active, lambda,
                                                           entropy,
                                                           cost_ratio/(1.0f + cost_ratio),
                                                           1.0f/(1.0f + cost_ratio),
                                                           n_positives, seed);
    (cpm->W).deserialize(&ss);
    
    if (ss.fail() | ss.eof() | ss.bad()) {
        throw std::runtime_error("Error when reading model file.");
    }
    
    return cpm;
}

std::pair<double, int> ConvexPolytopeMachine::predict(const SparseVector& s) {
    W.inner(s, score);
    
    int index = 0;
    double max_score = score[0];
    
    for (int i=1; i<k; ++i) {
        if (score[i] > max_score) {
            index = i;
            max_score = score[i];
        }
    }
    
    return std::make_pair(score[index], index);
}


std::pair<unsigned short, unsigned short> ConvexPolytopeMachine::heuristicMax(const SparseVector s, size_t cid) {
    // true argmax
    unsigned short true_imax = 0;
    double max_score = score[true_imax];
    
    for (unsigned short i = 1; i < k; ++i) {
        if (max_score < score[i]) {
            max_score = score[i];
            true_imax = i;
        }
    }
    
    double N = distinct_p;
    // not enough samples to compute an entropy score yet
    // just return argmax
    if (entropy <= 0 || N < k * 5.0f) {
        return std::make_pair(true_imax, true_imax);
    }
    
    // compute old and candidate entropy
    int old = assignments[cid];
    
    double h_old = 0;
    double h_new = 0;
    
    for (unsigned short i = 0; i < k; ++i) {
        double pi = occupancy[i]/N;
        double hpi = 0;
        
        if (pi > pepsilon) {
            hpi = -pi * std::log(pi);
        }
        h_old += hpi;
        
        if (old != -1) {
            if ((old == true_imax) || (i != old && i != true_imax)) {
                h_new += hpi;
            } else {
                if (i == old) {
                    double pold = (occupancy[i] - 1)/N;
                    h_new += -pold * std::log(pold);
                } else if (i == true_imax) {
                    double pnew = (occupancy[i] + 1)/N;
                    h_new += -pnew * std::log(pnew);
                }
            }
        } else {
            double pi;
            if (i == true_imax) {
                pi = (occupancy[i] + 1)/(N+1.0f);
            } else {
                pi = occupancy[i]/(N+1.0f);
            }
            
            h_new += -pi * std::log(pi);
        }
    }
    
    // return regular argmax when new entropy is large enough
    if ((h_new >= entropy) || (h_old < h_new)) {
        return std::make_pair(true_imax, true_imax);
    }
    
    // return argmax provided entropy is guaranteed to increase
    if (old != -1) {
        unsigned short imax = 0;
        double max_score = -std::numeric_limits<float>::infinity();

        for (unsigned short i = 0; i < k; ++i) {
            if (occupancy[i] < occupancy[old]) {
                if (max_score < score[i]) {
                    max_score = score[i];
                    imax = i;
                }
            }
        }
        
        return std::make_pair(imax, true_imax);
    } else {
        unsigned short imax = 0;
        double max_score = -std::numeric_limits<float>::infinity();
        
        for (unsigned short i = 0; i < k; ++i) {
            if (occupancy[i] < (k/N)) {
                if (max_score < score[i]) {
                    max_score = score[i];
                    imax = i;
                }
            }
        }
        
        return std::make_pair(imax, true_imax);
    }
}

void ConvexPolytopeMachine::setHistory(size_t cid, unsigned short imax) {
    int old = assignments[cid];
    if (cid >= n_positives){
        throw std::logic_error("positive instance special id >= number of positive instances.");
    }
    assignments[cid] = imax;
    
    occupancy[imax]++;
    if (old == -1) {
        distinct_p++;
    } else {
        occupancy[old]--;
    }
}

std::tuple<float, float, unsigned short> ConvexPolytopeMachine::oneStep(const std::tuple<int, const SparseVector, size_t>& lsi) {
    
    const double eta = 1.0/(lambda * (iter + 2.0)); // learning rate
    
    const SparseVector& s = std::get<1>(lsi);
    
    // get all scores
    W.inner(s, score);
    
    unsigned short imax;
    double max_score;
    
    double eloss = 0.0;
    
    // learn from instance
    if (std::get<0>(lsi) == outer_label) { // case y = +1
        
        // compute attribution
        auto imax_trueimax = heuristicMax(s, std::get<2>(lsi));
        imax = imax_trueimax.first;
        unsigned short true_imax = imax_trueimax.second;
        
        max_score = score[imax];
        
        // compute exclusion loss
        for (unsigned short i = 0; i < k; ++i) {
            if (i != imax) {
                eloss += std::max(0.0, score[i]);
            }
        }
        
        if (max_score < margin) {
            W.addInplace(s, eta * positive_cost, imax);
        }
        
        setHistory(std::get<2>(lsi), true_imax);
        imax = true_imax;
    } else { // case y = -1
        imax = 0;
        max_score = score[imax];
        
        // push down all classifiers as needed
        bool active= false;
        double* grad_mul = new double[k];
        
        for(unsigned short i = 0; i < k; ++i) {
            if (score[i] > -margin) {
                grad_mul[i] = -eta * negative_cost;
                active = true;
            } else {
                grad_mul[i] = 0.0;
            }
            
            if (score[i] > max_score) {
                imax = i;
                max_score = score[imax];
            }
        }
        
        if (active) W.addInplace(std::get<1>(lsi), grad_mul);
        delete[] grad_mul;
    }
    
    // L2 penalty
    double coeff = std::max(0.0, 1.0 - eta*lambda);
    W.mulInplace(coeff);
    
    iter++;
    return std::make_tuple(max_score, eloss, imax);
}
