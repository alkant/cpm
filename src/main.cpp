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

// main.cpp

// Author: Alex Kantchelian, 2014
// akant@cs.berkeley.edu

#include <iostream>
#include <unistd.h>
#include <stdlib.h>
#include <cmath>
#include <chrono>

#include <fstream>
#include <sstream>
#include <string>

#include "time.h"

#include "option_parser.h"
#include "sparse_vector.h"
#include "stochastic_data_adaptor.h"
#include "convex_polytope_machine.h"
#include "eval_utils.h"
#include "cpm.h"

int main(int argc, char* const argv[]) {
    OptionParser op("Perform CPM training and/or inference.");
    
    op.addOption("be quiet.", 'q', "quiet", true, false);
    
    op.addOption("number of classifiers.", 'k', "classifiers", true, (int) 1, nullptr);
    op.addOption("C regularization factor.", 'C', "C", true, 1.0f, nullptr);
    op.addOption("cost ratio of negatives vs positives.", '\0', "cost_ratio", true, 1.0f, nullptr);
    op.addOption("minimal (exp of) entropy to maintain in heuristic max. Value between 1 and k.", '\0',
                 "entropy", true, 1.0f, nullptr);
    
    size_t seed = (size_t) std::chrono::system_clock::now().time_since_epoch().count();
    op.addOption("random seed (for reproducibility).", '\0', "seed", false, seed, nullptr);
    
    // op.addOption("compute aggregated metrics instead of raw scores.", '\0', "agg_scores", true, false);
    
    op.addOption("outer class label (the class that will be decomposed).", '\0', "outer_label", true, (int) 1, nullptr);

    op.addOption("shuffle training set between epochs.", '\0', "reshuffle", true, false);
    
    op.addOption("number of iterations.", 'i',
                 "iterations", true, (int) 50000000, nullptr);
    
    op.addOption("train data file.", 't', "train", false, "", nullptr);
    op.addOption("test data file.", 'c', "test", false, "", nullptr);
    op.addOption("model in file. Will be ignored if in training mode.", 'm', "model_in", false, "", nullptr);
    op.addOption("model out file.", 'o', "model_out", false, "", nullptr);
    op.addOption("scores file.", 's', "scores", false, "", nullptr);
    
    op.parseCmdString(argc, argv);
    
    const bool verbose = !op.getBool("quiet");
    const int outer_label = op.getInt("outer_label");
    const char* trainfile = op.getString("train");
    const char* model_in = op.getString("model_in");
    const char* testfile = op.getString("test");
    const char* scoresfile = op.getString("scores");
    const int k = op.getInt("classifiers");
    const float C = op.getFloat("C");
    const int iterations = op.getInt("iterations");
    const float cost_ratio = op.getFloat("cost_ratio");
    const float entropy = op.getFloat("entropy");
    bool reshuffle = op.getBool("reshuffle");
    
    seed = op.getSizet("seed");
    if (sizeof(seed) == 8) {
        seed = seed ^ (seed >> 32);
    }
    
    CPM* model = nullptr;
    
    if (std::strlen(trainfile) > 0) {
        clock_t start_time = clock();
        
        StochasticDataAdaptor trainset(trainfile);
        
        clock_t end_time = clock();
        std::cout << "Loaded data in "
        << ((float) (end_time-start_time))/CLOCKS_PER_SEC << "s.\n";
        start_time = end_time;
        
        // train cpm
        model = new CPM(k, outer_label, 1.0f/C, entropy, cost_ratio, (unsigned short) seed);
        model->fit(trainset, iterations, reshuffle, verbose);
        
        end_time = clock();
        std::cout << "\nFinished " << iterations << " iterations in " << ((float) (end_time-start_time))/CLOCKS_PER_SEC << "s.\n";
        
        const char* model_out = op.getString("model_out");
        
        if(verbose && (std::strlen(model_out) > 0)) std::cout << "Writing model to " << model_out << '\n';
        
        if (std::strlen(model_out) > 0) {
            model->serializeModel(model_out);
        }
        
    } else if (std::strlen(model_in) > 0) {
        if(verbose) std::cout << "Reading model from " << model_in << '\n';
        model = CPM::deserializeModel(model_in);
    }
    
    if (model && std::strlen(testfile) > 0) {
        if (std::strlen(scoresfile) == 0) {
            std::cerr << "Missing output scores file.\n";
            exit(1);
        }
        
        StochasticDataAdaptor testset(testfile);
        
        std::ofstream rfile(scoresfile);
        
        for(size_t i = 0; i < testset.getNInstances(); ++i) {
            const std::tuple<int, SparseVector, size_t>& lic = testset.getInstance(i);
            
            auto score_sub = model->predict(std::get<1>(lic));
            
            // format: raw score (margin), assigned classifier, ground truth (model_outer_label == instance_label)
            rfile << score_sub.first << '\t' << score_sub.second << '\t' << (std::get<0>(lic) == model->outer_label) << '\n';
        }
    }
    
    delete model;
}
