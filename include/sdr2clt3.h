#ifndef SDR2CLT3_H
#define SDR2CLT3_H

#define __CL_ENABLE_EXCEPTIONS

#include <estmat.h>

#include <stdio.h>
#include <CL/cl.hpp>


class SDR2CL: public Minimizer{
private:
  int nPlanes, nTracks, nGPUThreads;
  bool readTracks;

  float **measX, **measY;
  float *chi2x, *chi2y;
  
  cl::Buffer bufx, bufdx, bufxx, bufxdx, bufdxdx, bufchi2x;
  cl::Buffer bufy, bufdy, bufyy, bufydy, bufdydy, bufchi2y;
  cl::Buffer bufmeasX, bufmeasY;
  
  float resVarX, resVarY; //Counters for threads
  
  void startReduction(int nThreads);
  void threadReduce(int min, int max);
  void threadTally();

  void runKernel(cl::Kernel& kernel, float resx, float resy, float scattervar, float dz);
  void copyMeasurements(float* measx, float *measy);
  
#ifdef DOTHREAD
  boost::mutex reduceGurad;
  boost::thread_group threads;
#endif

  public:
  SDR2CL(EstMat& mat, int nPlanes, int nTracks);
  ~SDR2CL();
  virtual void init();
  void operator() (size_t offset, size_t /*stride*/);
};

#endif
