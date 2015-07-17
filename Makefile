CXX          = g++
#CXXFLAGS     = -g -fPIC -msse4 -O3 -I/usr/include/eigen3 -I./include -lgsl -lgslcblas -lboost_system -lboost_thread -DNDEBUG -DDOTHREAD -std=c++11
CXXFLAGS     = -g -fPIC -msse4 -O3 -I/usr/include/eigen3 -I./include -lgsl -lgslcblas -lboost_system -lboost_thread -DNDEBUG -std=c++11 -Wall

LD           = g++
LDFLAGS      = -g -L.

ROOTCONFIG   := $(shell which root-config)

ROOTCXXFLAGS := $(shell $(ROOTCONFIG) --cflags)
ROOTLDFLAGS  := $(shell $(ROOTCONFIG) --ldflags)
ROOTLIBS     := $(shell $(ROOTCONFIG) --libs)
LIBS         := $(USERLIBS) $(ROOTLIBS) $(LIBS)

INCLUDEDIRS  := -
CXXFLAGS     := $(ROOTCXXFLAGS) $(CXXFLAGS)
LDFLAGS      := $(ROOTLDFLAGS) $(LDFLAGS)

simple: simple.cc
	$(LD) $(LDFLAGS) $(CXXFLAGS) $(LIBS) -o $@ $^ 

multitrack: multitrack.cc
	$(LD) $(LDFLAGS) $(CXXFLAGS) $(LIBS) -o $@ $^ 

estmat: estmatapp.cc ./src/estmat.cc ./src/sdr2clt3.cpp
	$(LD) $(LDFLAGS) $(CXXFLAGS) $(LIBS) -lOpenCL -o $@ $^ 

scanalign: scanalign.cc ./src/estmat.cc ./src/sdr2clt3.cpp
	$(LD) $(LDFLAGS) $(CXXFLAGS) $(LIBS) -lOpenCL -o $@ $^ 

estmat2: estmatapp2.cc ./src/estmat.cc
	$(LD) $(LDFLAGS) $(CXXFLAGS) $(LIBS) -o $@ $^ 

.PHONY: noisesim
noisesim: noisesimkf noisesimdaf noisesimclu

noisesimkf:  noisesim.cc
	$(LD) $(LDFLAGS) $(CXXFLAGS) $(LIBS) -o $@ $^ 

noisesimdaf:  noisesim.cc
	$(LD) $(LDFLAGS) $(CXXFLAGS) -DDAF $(LIBS) -o $@ $^ 

noisesimclu:  noisesim.cc
	$(LD) $(LDFLAGS) $(CXXFLAGS) -DCLU $(LIBS) -o $@ $^ 
