#ifndef SDR2CLT3_H
#define SDR2CLT3_H

#define __CL_ENABLE_EXCEPTIONS

#include <estmat.h>

#include <stdio.h>
#include <CL/cl.hpp>


class SDR2CL: public Minimizer{
private:
  int nPlanes, nTracks, nParallel;
  int nsplit, nreduced;
  bool readTracks;

  int nCudaCores;
  int workGroupSize;
  int nReduceThreads;
 

  float **measX;
  float *measVar, *invScatter, *dz;
  float **chi2f, **chi2b;
  
  void writeSysParams();
  void reduceStage2(cl::Buffer& buf, float* chi2);
  
#ifdef DOTHREAD
  boost::mutex reduceGurad;
  boost::thread_group threads;
#endif

  //opencl stuff
  std::vector<cl::Platform> platforms;
  std::vector<cl::Device> devices;
  cl::Context context;
  cl::CommandQueue queue;
  
  cl::Program program;
  cl::Kernel fitPlanes;
  cl::Kernel reduce;

  std::vector<cl::Buffer> measbufs;
  std::vector<cl::Buffer> chi2fbufs;
  std::vector<cl::Buffer> chi2bbufs;
  std::vector<cl::Buffer> reducedxbufs;
  std::vector<cl::Buffer> reducedybufs;
  cl::Buffer measvarbuf, invscatterbuf, dzbuf;

  public:
  SDR2CL(EstMat& mat, int nPlanes, int nTracks);
  virtual ~SDR2CL();
  virtual void init();
  void operator() (size_t offset, size_t /*stride*/);
};

#endif
