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

// option_parser.h

// Author: Alex Kantchelian, 2014
// akant@cs.berkeley.edu

#ifndef __cpm__option_parser__
#define __cpm__option_parser__

#include <iostream>
#include <map>
#include <vector>
#include <string>

// Hacky command line option parser so we don't have to import Boost's one.

template <class T> class Option {
public:
    Option(const char* desc, const char short_name,
           const char* long_name, bool provide_default, T value, const std::vector<T>* allowed) : desc(desc), short_name(short_name),
            long_name(long_name), allowed(allowed), provide_default(provide_default){
        
        this->value = new T(value);
    }
    
    Option(Option&& other) : desc(other.desc), short_name(other.short_name),
    long_name(other.long_name), allowed(other.allowed), provide_default(other.provide_default), value(other.value){
        other.value = nullptr;
        other.allowed = nullptr;
    }
    
    ~Option() {
        delete value;
    }
    
    void parseValue(const char* string);
    void show();
    void setValue(T val);
    
    const char* desc;
    const char short_name;
    const char* long_name;
    const std::vector<T>* allowed;
    const bool provide_default;
    
    T* value;
};

class OptionParser {
public:
    OptionParser(const char* desc);
    
    void addOption(const char* desc, const char short_name,
                   const char* long_name, bool provide_default, bool default_value);
    void addOption(const char* desc, const char short_name,
                   const char* long_name, bool provide_default, int default_value, const std::vector<int>* allowed);
    void addOption(const char* desc, const char short_name,
                   const char* long_name, bool provide_default, float default_value, const std::vector<float>* allowed);
    void addOption(const char* desc, const char short_name,
                   const char* long_name, bool provide_default, size_t default_value, const std::vector<size_t>* allowed);
    void addOption(const char* desc, const char short_name,
                   const char* long_name, bool provide_default, const char* default_value, const std::vector<const char*>* allowed);
    
    void parseCmdString(int argc, char* const argv[]);
    
    bool getBool(const char* long_name);
    int getInt(const char* long_name);
    size_t getSizet(const char* long_name);
    float getFloat(const char* long_name);
    const char* getString(const char* long_name);
    
    void showHelp();

private:
    void terminate(const char* msg);
    
    int searchAndFill(int argc, char* const argv[], int pos);
    
    const char* desc;
    std::vector<Option<bool>> bool_options;
    std::vector<Option<int>> int_options;
    std::vector<Option<size_t>> sizet_options;
    std::vector<Option<const char*>> string_options;
    std::vector<Option<float>> float_options;
};

#endif /* defined(__cpm__option_parser__) */
