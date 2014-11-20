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

// eval_utils.cpp

// Author: Alex Kantchelian, 2014
// akant@cs.berkeley.edu

#include <limits>
#include <algorithm>
#include <cmath>

#include <stdexcept>

#include "eval_utils.h"

namespace evalutils {
    const float margin = 1.0f;
    
    double entropy(const int* assignments, size_t length, unsigned short k) {
        int* occ = new int[k]();
        
        for (size_t i = 0; i < length; ++i) {
            if ((assignments[i] < 0) || (assignments[i] >= k)) {
                throw std::logic_error("Assignment int outside of [0;k-1].");
            }
            occ[assignments[i]]++;
        }
        
        double entropy = 0;
        
        for (unsigned short i = 0; i < k; ++i) {
            double p = ((double) occ[i])/length;
            if (p > 0) {
                entropy -= p * std::log(p);
            }
        }
        
        entropy /= std::log(2.0);
        
        delete[] occ;
        return entropy;
    }
    
    std::unique_ptr<std::map<Metric, double>> measure(const StochasticDataAdaptor& testset, ConvexPolytopeMachine& model) {
        int outer_label = model.outer_label;
        int k = model.k;
        
        double cost_pos = 0;
        double cost_neg = 0;
        double cost_exclusion = 0;
        double entropy = 0;
        double l2 = (model.getW()).l2norm();
        
        int* occ = new int[k]();
        float* p = new float[k]();
        
        int n_neg = 0;
        int n_pos = 0;
        
        size_t fps = 0;
        size_t fns = 0;
        auto all_scores = new std::vector<std::pair<bool, float>>();
        
        for (size_t instance=0; instance<testset.getNInstances(); ++instance) {
            auto lic = testset.getInstance(instance);
            const SparseVector sv = std::get<1>(lic);
            
            auto score_index = model.predict(sv);
            float score = (float) score_index.first;
            int index = score_index.second;
            bool pred = score > 0.0f;
            
            const double* scores = model.getScores();
            
            if (std::get<0>(lic) == outer_label) { // positive sample
                occ[index] += 1;
                
                if(score < margin) {
                    cost_pos += margin - score;
                }
                
                for(int i = 0; i < k; i++) {
                    if (i != index) {
                        cost_exclusion += (scores[i] > 0.0) ? scores[i] : 0.0;
                    }
                }
                
                if(score < 0) {
                    fns++;
                    if (pred) {
                        throw std::logic_error("Negative score but predicted positive (pos instance).");
                    }
                } else {
                    if (!pred) {
                        throw std::logic_error("Positive score but predicted negative (pos instance).");
                    }
                }
                
                all_scores->emplace_back(true, score);
                
                n_pos++;
            } else { // negative sample
                for(int i = 0; i < k; i++) {
                    cost_neg += (scores[i] > -margin) ? (margin + scores[i]) : 0.0;
                }
                
                if(score >= 0) {
                    fps++;
                    if(!pred) {
                        throw std::logic_error("Positive score but predicted negative (neg instance).");
                    }
                } else if (pred) {
                    throw std::logic_error("Negative score but predicted positive (neg instance).");
                }
                
                all_scores->emplace_back(false, score);
                
                n_neg++;
            }
        }
        
        // compute entropy
        for (int i = 0; i < k; i++) {
            p[i] = ((float) occ[i]) / ((float) n_pos);
            if(p[i] > 0.0) {
                entropy -= p[i] * std::log(p[i]);
            }
        }
        entropy /= std::log(2.0f); // entropy in bits
        
        cost_exclusion /= n_pos;
        
        double misc_cost = (cost_neg + cost_pos)/(n_pos + n_neg);
        
        cost_pos /= n_pos;
        cost_neg /= n_neg;
        
        double accuracy = 1.0 - ((double) (fps + fns)) / (n_pos + n_neg);
        size_t tps = (size_t) n_pos - fns;
        double tpr = ((double) tps) / n_pos;
        double fpr = ((double) fps) / n_neg;
        double precision = ((double) tps) / (tps + fps);
        
        // compute AUCs
        std::sort(all_scores->begin(), all_scores->end(),
                  [](const std::pair<bool, float>& lhs, const std::pair<bool, float>& rhs) {
            return lhs.second > rhs.second;}); // sorted by decreasing scores
        
        size_t i = 0;
        size_t fn = (size_t) n_pos;
        size_t fp = 0;
        size_t top_correct = 0;
        
        double tprs = ((float) fp) / n_neg;
        double fprs = 1.0f - ((float) fn) / ((float) n_pos);
        double last_tprs, last_fprs;
        
        double area001 = 0.0;
        double area01 = 0.0;
        double area1 = 0.0;
        while (i < all_scores->size()) {
            last_tprs = tprs;
            last_fprs = fprs;
            
            if ((*all_scores)[i].first) {
                fn -= 1;
                if (fp == 0) top_correct += 1;
            } else {
                fp += 1;
            }
            i += 1;
            
            while (i < all_scores->size() &&
                   (*all_scores)[i-1].second == (*all_scores)[i].second) {
                if ((*all_scores)[i].first) {
                    fn -= 1;
                    if (fp == 0) top_correct += 1;
                } else {
                    fp += 1;
                }
                i += 1;
            }
            
            fprs = ((double) fp)/n_neg;
            tprs = 1.0 - ((double) fn)/n_pos;
            
            double darea1 = (fprs - last_fprs) * (last_tprs + tprs)/2.0;
            
            area1 += darea1;
            
            if (last_fprs < .1) {
                double darea01 = 0.0;
                if (fprs <= .1) {
                    darea01 = (fprs - last_fprs) * (last_tprs + tprs)/2.0;
                } else {
                    if (fprs > last_fprs) {
                        double tprs01 = last_tprs + (.1 - last_fprs)/(fprs - last_fprs) * (tprs - last_tprs);
                        darea01 = (.1 - last_fprs) * (last_tprs + tprs01)/2.0;
                    }
                }
                
                area01 += darea01;
                
                if (last_fprs < .01) {
                    double darea001 = 0.0;
                    if (fprs <= .01) {
                        darea001 = (fprs - last_fprs) * (last_tprs + tprs)/2.0;
                    } else {
                        if (fprs > last_fprs) {
                            double tprs001 = last_tprs + (.01 - last_fprs)/(fprs - last_fprs) * (tprs - last_tprs);
                            darea001 = (.01 - last_fprs) * (last_tprs + tprs001)/2.0;
                        }
                    }
                    
                    area001 += darea001;
                }
            }
        }
        
        area01 *= 10;
        area001 *= 100;
        double absolute_top = ((double) top_correct) / n_pos;
        
        delete[] occ;
        delete all_scores;
        delete[] p;
        
        auto res = std::unique_ptr<std::map<Metric, double>>(new std::map<Metric, double>());
        (*res)[Metric::Cost] = misc_cost;
        (*res)[Metric::CostPositives] = cost_pos;
        (*res)[Metric::CostNegatives] = cost_neg;
        (*res)[Metric::L2] = l2;
        
        (*res)[Metric::Redundancy] = cost_exclusion;
        (*res)[Metric::Entropy] = entropy;
        
        (*res)[Metric::Accuracy] = accuracy;
        (*res)[Metric::TruePositiveRate] = tpr;
        (*res)[Metric::FalsePositiveRate] = fpr;
        (*res)[Metric::Precision] = precision;
        
        (*res)[Metric::AUC] = area1;
        (*res)[Metric::AUC01] = area01;
        (*res)[Metric::AUC001] = area001;
        (*res)[Metric::AbsoluteTop] = absolute_top;
        return res;
    }
}
