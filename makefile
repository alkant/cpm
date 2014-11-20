CXX=g++
CXXFLAGS=-Wall -pedantic -pthread -std=c++11
SWIGFLAGS=-c++ -python -O -builtin
OFLAG=-O3
VPATH=src
OBJDIR=build
BINDIR=bin

all: directories build cmdapp

build: $(OBJDIR)/sparse_vector.o $(OBJDIR)/dense_matrix.o \
			 $(OBJDIR)/stochastic_data_adaptor.o \
			 $(OBJDIR)/convex_polytope_machine.o \
			 $(OBJDIR)/eval_utils.o \
			 $(OBJDIR)/option_parser.o \
			 $(OBJDIR)/parallel_eval.o \
			 $(OBJDIR)/cpm.o

cmdapp: $(BINDIR)/cpm

$(BINDIR)/cpm: $(OBJDIR)/main.o $(OBJDIR)/sparse_vector.o \
			 $(OBJDIR)/dense_matrix.o \
			 $(OBJDIR)/stochastic_data_adaptor.o \
			 $(OBJDIR)/eval_utils.o \
			 $(OBJDIR)/convex_polytope_machine.o\
			 $(OBJDIR)/option_parser.o \
			 $(OBJDIR)/parallel_eval.o \
			 $(OBJDIR)/cpm.o
	$(CXX) -o $@ $(OFLAG) $(CXXFLAGS) $^

wrapper: python.i
	swig $(SWIGFLAGS) -outdir $(VPATH) -o $(VPATH)/python_wrap.cpp $^

directories:
	mkdir -p $(OBJDIR)
	mkdir -p $(BINDIR)

$(OBJDIR)/sparse_vector.o: sparse_vector.cpp sparse_vector.h
	$(CXX) -c $(CXXFLAGS) $(OFLAG) $< -o $@

$(OBJDIR)/dense_matrix.o: dense_matrix.cpp \
	dense_matrix.h
	$(CXX) -c $(CXXFLAGS) $(OFLAG) $< -o $@

$(OBJDIR)/parallel_eval.o: parallel_eval.cpp parallel_eval.h
	$(CXX) -c $(CXXFLAGS) $(OFLAG) $< -o $@

$(OBJDIR)/stochastic_data_adaptor.o: stochastic_data_adaptor.cpp \
	stochastic_data_adaptor.h
	$(CXX) -c $(CXXFLAGS) $(OFLAG) $< -o $@

$(OBJDIR)/convex_polytope_machine.o: convex_polytope_machine.cpp \
	convex_polytope_machine.h
	$(CXX) -c $(CXXFLAGS) $(OFLAG) $< -o $@

$(OBJDIR)/option_parser.o: option_parser.cpp option_parser.h
	$(CXX) -c $(CXXFLAGS) $(OFLAG) $< -o $@

$(OBJDIR)/cpm.o: cpm.cpp cpm.h
	$(CXX) -c $(CXXFLAGS) $(OFLAG) $< -o $@

$(OBJDIR)/eval_utils.o: eval_utils.cpp eval_utils.h
	$(CXX) -c $(CXXFLAGS) $(OFLAG) $< -o $@

$(OBJDIR)/main.o: main.cpp
	$(CXX) -c $(CXXFLAGS) $(OFLAG) $< -o $@

clean:
	rm -rf $(OBJDIR)/*
	rm -f $(BINDIR)/*
