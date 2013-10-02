CXX          = g++
#CXXFLAGS     = -g -fPIC -msse4 -O3 -I/usr/include/eigen2 -I./include -lgsl -lgslcblas -lboost_system -lboost_thread -DNDEBUG -DDOTHREAD
#CXXFLAGS     = -g -fPIC -msse4 -O3 -I/usr/include/eigen2 -I./include -lgsl -lgslcblas -lboost_system -lboost_thread -DNDEBUG
CXXFLAGS     = -g -fPIC -msse4 -O3 -I/usr/include/eigen2 -I./include -lgsl -lgslcblas -lboost_system -lboost_thread

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

noisesim: noisesim.cc
	$(LD) $(LDFLAGS) $(CXXFLAGS) $(LIBS) -o $@ $^ 

estmat: estmatapp.cc ./src/estmat.cc 
	$(LD) $(LDFLAGS) $(CXXFLAGS) $(LIBS) -o $@ $^ 
