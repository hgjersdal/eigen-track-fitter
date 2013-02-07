CXX          = g++
CXXFLAGS     = -g -fPIC -msse3 -O3 -I/usr/include/eigen2 -I./include -lgsl -lgslcblas -DNDEBUG
#CXXFLAGS     = -g -fPIC -msse3 -O0 -I/usr/include/eigen2 -I./ -lgsl -lgslcblas 
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

estmat: estmatapp.cc ./src/estmat.cc 
	$(LD) $(LDFLAGS) $(CXXFLAGS) $(LIBS) -DMSE1=1 -o $@ $^ 
