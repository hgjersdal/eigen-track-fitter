#ifndef SDR2CLT3_H
#define SDR2CLT3_H

#define __CL_ENABLE_EXCEPTIONS

#include <estmat.h>

#include <stdio.h>
#include <CL/cl.hpp>


class SDR2CL: public Minimizer{
private:
  int nPlanes, nTracks;
  bool readTracks;

  cl::Context context;
  cl::CommandQueue queue;
  
  cl::Program program;
  cl::Kernel firstFW, secondFW, restFW;
  cl::Kernel firstBW, secondBW, restBW;
  
  float **measX, **measY;
  float *x, *dx, *xx, *xdx, *dxdx, *chi2x;
  float *y, *dy, *yy, *ydy, *dydy, *chi2y;
  
  cl::Buffer bufx, bufdx, bufxx, bufxdx, bufdxdx, bufchi2x;
  cl::Buffer bufy, bufdy, bufyy, bufydy, bufdydy, bufchi2y;
  cl::Buffer bufmeasX, bufmeasY;
  
  public:
  SDR2CL(EstMat& mat, int nPlanes, int nTracks);
  ~SDR2CL();
  virtual void init();
  void prepareForFit();
  void runKernel(cl::Kernel& kernel, float resx, float resy, float scattervar, float dz);
  void copyMeasurements(float* measx, float *measy);
  void operator() (size_t offset, size_t /*stride*/);
};

#endif
