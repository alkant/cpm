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

// convex_polytope_machine.h

// Author: Alex Kantchelian, 2014
// akant@cs.berkeley.edu

#ifndef __cpm__convex_polytope_machine__
#define __cpm__convex_polytope_machine__

#include <iostream>
#include <random>
#include <utility>

#include "sparse_vector.h"
#include "dense_matrix.h"

class ConvexPolytopeMachine{
public:
    /* Initalizes the parameters of the SGD
     *
     * outer_label: label on the outside of the polytope
     * dim: maximum number of dimensions
     * k: number of sub-classifiers
     * lambda: L2 penalty, taken per iteration
     * entropy: minimum assignment entropy to maintain
     * negative_cost: cost incurred on false positives
     * positive_cost: cost incurred on true positives
     * n_positives: total number of outer_label samples
     */
    ConvexPolytopeMachine(int outer_label, int dim, unsigned short k,
                          float lambda, float entropy,
                          float negative_cost, float positive_cost, size_t n_positives,
                          unsigned int seed);
    
    // destructor
    ~ConvexPolytopeMachine() {
        delete[] score;
        delete[] assignments;
        delete[] occupancy;
    }
    
    // perform one SGD step with the given sample
    std::tuple<float, float, unsigned short> oneStep(const std::tuple<int, const SparseVector, size_t>& lsi);
    
    // get number of iterations since beginning
    size_t getIter() const {return iter;};
    
    // get dense matrix
    const DenseMatrix& getW() const {
        return W;
    }
        
    // get score and assigned classifier for given instance
    std::pair<double, int> predict(const SparseVector& s);
    
    // scores for each sub-classifier
    const double* getScores() const {return score;}
    
    // clear W and set iter to 0
    void clear();
    
    // get table of all assignments of positive class
    const int* getAssignments() const {return assignments;};
    
    const int getAssignment(size_t cid) const {return assignments[cid];}
    
    // write model to disk
    void serializeModel(const char* filename) const;
    
    // read model from disk
    static ConvexPolytopeMachine* deserializeModel(const char* filename);
    
    // margin value
    const float margin = 1.0f;
    
    const int outer_label;
    const unsigned short k;
    const float lambda;
    const float entropy;
    const float negative_cost;
    const float positive_cost;
    const size_t n_positives;
    const unsigned int seed;

private:
    const float pepsilon = 1e-6f;
    double* score; // w's
    size_t iter;
    DenseMatrix W;
    
    int* assignments; // holds assignments history for outer instances
    unsigned int* occupancy; // holds # of firings per classifier
    size_t distinct_p; // number of entries filled up in history
    
    void setHistory(size_t cid, unsigned short imax);
    std::pair<unsigned short, unsigned short> heuristicMax(const SparseVector s, size_t cid);
    // computes optimal assignment that will maintain entropy constraint
    // updates all counting-related fields (namely history and occupancy)
};

#endif /* defined(__cpm__convex_polytope_machine__) */
