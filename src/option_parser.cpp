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

// option_parser.cpp

// Author: Alex Kantchelian, 2014
// akant@cs.berkeley.edu

#include <typeinfo>
#include <cstring>

#include "option_parser.h"

template <class T> void Option<T>::show(){
    if (long_name) {
        std::cout << "--" << long_name << " ";
    }
    if (short_name != '\0') {
        std::cout << '-' << short_name << " ";
    }
    
    if (typeid(value) == typeid(int*)) {
        std::cout << "<int> ";
    }
    
    if (typeid(value) == typeid(float*)) {
        std::cout << "<float> ";
    }
    
    if (typeid(value) == typeid(size_t*)) {
        std::cout << "<unsigned long> ";
    }
    
    if (typeid(value) == typeid(const char**)) {
        std::cout << "<string> ";
    }
    
    std::cout << "  " << desc << '\n';
    
    if (allowed) {
        std::cout << "    Allowed: {";
        
        if (typeid(value) == typeid(int*)) {
            for(int val: *(std::vector<int>*) allowed) {
                std::cout << val<< ", ";
            }
        }
        
        if (typeid(value) == typeid(float*)) {
            for(float val: * (std::vector<float>*) allowed) {
                std::cout << val<< ", ";
            }
        }
        
        if (typeid(value) == typeid(size_t*)) {
            for(size_t val: * (std::vector<size_t>*) allowed) {
                std::cout << val<< ", ";
            }
        }
        
        if (typeid(value) == typeid(const char**)) {
            for(const char* val: * (std::vector<const char*>*) allowed) {
                std::cout << val<< ", ";
            }
        }
        
        std::cout << "}\n";
    }
    
    if (provide_default) {
        if (typeid(value) == typeid(bool*)) {
            std::cout << "    Default: " << (* (bool*) value ? "True" : "False") << '\n';
            return;
        }
        
        if (typeid(value) == typeid(int*)) {
            std::cout << "    Default: " << * (int*)value << '\n';
            return;
        }
        
        if (typeid(value) == typeid(float*)) {
            std::cout << "    Default: " << * (float*)value << '\n';
            return;
        }
        
        if (typeid(value) == typeid(size_t*)) {
            std::cout << "    Default: " << * (size_t*)value << '\n';
            return;
        }
        
        if (typeid(value) == typeid(const char**)) {
            std::cout << "    Default: " << * (const char**)value << '\n';
            return;
        }
    }
}

template <> void Option<bool>::setValue(bool val) {
    *value = val;
}

template <> void Option<int>::setValue(int val) {
    if (allowed) {
        bool allow = false;
        
        for (auto allo : *allowed) {
            if (allo == val) {
                allow = true;
                break;
            }
        }
        
        if (!allow) {
            std::cerr << "Value " << val << " not allowed for option " << long_name << '\n';
            exit(1);
        }
    }
    
    *value = val;
}

template <> void Option<float>::setValue(float val) {
    if (allowed) {
        bool allow = false;
        
        for (auto allo : *allowed) {
            if (allo == val) {
                allow = true;
                break;
            }
        }
        
        if (!allow) {
            std::cerr << "Value " << val << " not allowed for option " << long_name << '\n';
            exit(1);
        }
    }
    
    *value = val;
}

template <> void Option<size_t>::setValue(size_t val) {
    if (allowed) {
        bool allow = false;
        
        for (auto allo : *allowed) {
            if (allo == val) {
                allow = true;
                break;
            }
        }
        
        if (!allow) {
            std::cerr << "Value " << val << " not allowed for option " << long_name << '\n';
            exit(1);
        }
    }
    
    *value = val;
}

template <> void Option<const char*>::setValue(const char* val) {
    if (allowed) {
        bool allow = false;
        
        for (auto allo : *allowed) {
            if (0 == std::strcmp(allo, val)) {
                allow = true;
                break;
            }
        }
        
        if (!allow) {
            std::cerr << "Value " << val << " not allowed for option " << long_name << '\n';
            exit(1);
        }
    }
    
    *value = val;
}

OptionParser::OptionParser(const char* desc) : desc(desc) {}

void OptionParser::addOption(const char *desc, const char short_name, const char *long_name, bool provide_default, bool default_value) {
    bool_options.emplace_back(desc, short_name, long_name, provide_default, default_value, nullptr);
}

void OptionParser::addOption(const char *desc, const char short_name, const char *long_name, bool provide_default, int default_value, const std::vector<int>* allowed) {
    int_options.emplace_back(desc, short_name, long_name, provide_default, default_value, allowed);
}

void OptionParser::addOption(const char *desc, const char short_name, const char *long_name, bool provide_default, float default_value, const std::vector<float>* allowed) {
    float_options.emplace_back(desc, short_name, long_name, provide_default, default_value, allowed);
}

void OptionParser::addOption(const char *desc, const char short_name, const char *long_name, bool provide_default, size_t default_value, const std::vector<size_t>* allowed) {
    sizet_options.emplace_back(desc, short_name, long_name, provide_default, default_value, allowed);
}

void OptionParser::addOption(const char *desc, const char short_name, const char *long_name, bool provide_default, const char* default_value, const std::vector<const char*>* allowed) {
    string_options.emplace_back(desc, short_name, long_name, provide_default, default_value, allowed);
}

void OptionParser::showHelp() {
    std::cout << desc << "\n\n";
    
    for (auto& opt: bool_options) {
        opt.show();
    }
    
    for (auto& opt: int_options) {
        opt.show();
    }
    
    for (auto& opt: float_options) {
        opt.show();
    }
    
    for (auto& opt: sizet_options) {
        opt.show();
    }
    
    for (auto& opt: string_options) {
        opt.show();
    }
}

void OptionParser::terminate(const char* msg) {
    std::cerr << msg << '\n';
    exit(1);
}

void OptionParser::parseCmdString(int argc, char* const argv[]) {
    for (int i = 1; i < argc; ) {
        if (argv[i][0] == '-') {
            if (0 == std::strcmp("--help", argv[i]) || 0 == std::strcmp("-h", argv[i])) {
                showHelp();
                exit(0);
            }
            
            i = searchAndFill(argc, argv, i);
        } else {
            terminate("Positional arguments not allowed");
        }
    }
}

int OptionParser::searchAndFill(int argc, char * const argv[], int pos){
    //check if switch is well formed
    size_t size = std::strlen(argv[pos]);
    if (size < 2) {
        std::cerr << "Invalid switch " << argv[pos] << '\n';
        exit(1);
    }
    
    if (size > 2 && (argv[pos][1] != '-')) {
        std::cerr << "Invalid switch " << argv[pos] << '\n';
        exit(1);
    }
    
    for (auto& opt: bool_options) {
        if (argv[pos][1] == '-' ? 0 == std::strcmp(opt.long_name, (argv[pos] + 2)) : argv[pos][1] == opt.short_name) {
            opt.setValue(true);
            return pos + 1;
        }
    }
    
    for (auto& opt: int_options) {
        if (argv[pos][1] == '-' ? 0 == std::strcmp(opt.long_name, (argv[pos] + 2)) : argv[pos][1] == opt.short_name) {
            pos++;
            if (pos >= argc) {
                std::cerr << "Option --" << opt.long_name << " requires an <int> argument.\n";
                exit(1);
            }
            opt.setValue((int) std::strtol(argv[pos], nullptr, 10));
            return pos + 1;
        }
    }
    
    for (auto& opt: float_options) {
        if (argv[pos][1] == '-' ? 0 == std::strcmp(opt.long_name, (argv[pos] + 2)) : argv[pos][1] == opt.short_name) {
            pos++;
            if (pos >= argc) {
                std::cerr << "Option --" << opt.long_name << " requires a <float> argument.\n";
                exit(1);
            }
            opt.setValue(std::strtof(argv[pos], nullptr));
            return pos + 1;
        }
    }
    
    for (auto& opt: sizet_options) {
        if (argv[pos][1] == '-' ? 0 == std::strcmp(opt.long_name, (argv[pos] + 2)) : argv[pos][1] == opt.short_name) {
            pos++;
            if (pos >= argc) {
                std::cerr << "Option --" << opt.long_name << " requires an <unsigned long> argument.\n";
                exit(1);
            }
            opt.setValue(std::strtoul(argv[pos], nullptr, 10));
            return pos + 1;
        }
    }
    
    for (auto& opt: string_options) {
        if (argv[pos][1] == '-' ? 0 == std::strcmp(opt.long_name, (argv[pos] + 2)) : argv[pos][1] == opt.short_name) {
            pos++;
            if (pos >= argc) {
                std::cerr << "Option --" << opt.long_name << " requires a <string> argument.\n";
                exit(1);
            }
            opt.setValue(argv[pos]);
            return pos + 1;
        }
    }
    
    std::cerr << "Invalid switch " << argv[pos] << '\n';
    exit(1);
}

bool OptionParser::getBool(const char* long_name) {
    for (auto& opt: bool_options) {
        if (0 == std::strcmp(opt.long_name, long_name)) {
            return *opt.value;
        }
    }
    
    std::cerr << "Option " << long_name << " is not defined for type <bool>.\n";
    exit(1);
}

int OptionParser::getInt(const char* long_name) {
    for (auto& opt: int_options) {
        if (0 == std::strcmp(opt.long_name, long_name)) {
            return *opt.value;
        }
    }
    
    std::cerr << "Option " << long_name << " is not defined for type <int>.\n";
    exit(1);
}

size_t OptionParser::getSizet(const char* long_name) {
    for (auto& opt: sizet_options) {
        if (0 == std::strcmp(opt.long_name, long_name)) {
            return *opt.value;
        }
    }
    
    std::cerr << "Option " << long_name << " is not defined for type <size_t>.\n";
    exit(1);
}

float OptionParser::getFloat(const char* long_name) {
    for (auto& opt: float_options) {
        if (0 == std::strcmp(opt.long_name, long_name)) {
            return *opt.value;
        }
    }
    
    std::cerr << "Option " << long_name << " is not defined for type <float>.\n";
    exit(1);
}

const char* OptionParser::getString(const char* long_name) {
    for (auto& opt: string_options) {
        if (0 == std::strcmp(opt.long_name, long_name)) {
            return *opt.value;
        }
    }
    
    std::cerr << "Option " << long_name << " is not defined for type <const char*>.\n";
    exit(1);
}
