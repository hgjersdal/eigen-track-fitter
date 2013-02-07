CXX          = g++
CXXFLAGS     = -g -fPIC -msse3 -O3 -I/usr/include/eigen2 -I./ -lgsl -lgslcblas -DNDEBUG
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

noisesimckf: noisesim.cc
	$(LD) $(LDFLAGS) $(CXXFLAGS) $(LIBS) -DCKF=1 -o $@ $^ 

noisesimckfdaf: noisesim.cc
	$(LD) $(LDFLAGS) $(CXXFLAGS) $(LIBS) -DCKFDAF=1 -o $@ $^ 

noisesim: noisesim.cc
	$(LD) $(LDFLAGS) $(CXXFLAGS) $(LIBS) -DDAF=1 -o $@ $^ 

simple: simple.cc
	$(LD) $(LDFLAGS) $(CXXFLAGS) $(LIBS) -o $@ $^ 

estmat: estmat.cc 
	$(LD) $(LDFLAGS) $(CXXFLAGS) $(LIBS) -DMSE1=1 -o $@ $^ 

