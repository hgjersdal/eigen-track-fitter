#include <utility>
#include <iostream>
#include <fstream>
#include <string>
#include <unistd.h>
#include "estmat.h"
#include "sdr2clt3.h"


SDR2CL::SDR2CL(EstMat& mat, int nplanes, int ntracks) : Minimizer(mat), nPlanes(nplanes), readTracks(false), nTracks(ntracks){
  // Get available platforms
  cl::Platform::get(&platforms);
  nParallel = 4; // is the n in floatn divided by 2. float8-> 4

  //adjust nTracks to n832
  int tmp = (nTracks/nParallel) / 32;
  cout << "Fitting " << tmp * nParallel * 32 << " tracks out of  " << nTracks  << endl;
  nTracks = tmp * 32 * nParallel;

  // Select the default platform and create a context using this platform and the GPU
  cl_context_properties cps[3] = { 
    CL_CONTEXT_PLATFORM, 
    (cl_context_properties)(platforms[0])(), 
    0 
  };
  try{
    context = cl::Context( CL_DEVICE_TYPE_GPU, cps);
  }  catch(cl::Error error) {
    std::cout << error.what() << "(" << error.err() << ")" << std::endl;
  }
  // Get a list of devices on this platform
  devices = context.getInfo<CL_CONTEXT_DEVICES>();
    
  // Create a command queue and use the first device
  queue = cl::CommandQueue(context, devices[0]);
    
    
  nCudaCores = 336; //Hard coded :( MAX_COMPUTE_UNITS * 8
  nCudaCores = 48; //Hard coded :( MAX_COMPUTE_UNITS * 8
  //Print some device info
  cout << "Device: " << endl;
  cout << "Name: " << devices[0].getInfo<CL_DEVICE_NAME>() << endl;
  cout << "CL_DEVICE_GLOBAL_MEM_SIZE: "<< devices[0].getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() << endl;
  cout << "CL_DEVICE_LOCAL_MEM_SIZE: "<< devices[0].getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() << endl;
  cout << "CL_DEVICE_MAX_COMPUTE_UNITS: "<< devices[0].getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << endl;
  cout << "Number of cores (hard coded) " << nCudaCores << endl;


  // Read source file
  std::ifstream sourceFile("src/openclkalmant5.cl");
  std::string sourceCode( std::istreambuf_iterator<char>(sourceFile),
			  (std::istreambuf_iterator<char>()));
  cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length()+1));
    
  // Make program of the source code in the context
  program = cl::Program(context, source);
    
  // Build program for the device. Print log if it fails.
  try{
    program.build(devices);
  } catch(cl::Error error) {
    std::cout << "Build failed!" << std::endl;
    std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(devices[0]) << std::endl;
    std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(devices[0]) << std::endl;
    std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]) << std::endl;
    exit(1);
  }

  //Kernels
  fitPlanes = cl::Kernel(program, "fitPlanes");
  reduce = cl::Kernel(program, "reduce");

  workGroupSize = reduce.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(devices[0]);
  nReduceThreads = nCudaCores * workGroupSize;
  cout << "Work group size " << workGroupSize << endl;

  // Create opencl buffers 
  //read-write
  for(int ii = 0; ii < nPlanes; ii++){
    measbufs.push_back(cl::Buffer(context, CL_MEM_READ_WRITE, 2 * nTracks * sizeof(float)));
  }
  for(int ii = 0; ii < nPlanes - 2; ii++){
    chi2fbufs.push_back(cl::Buffer(context, CL_MEM_READ_WRITE, 2 * nTracks * sizeof(float)));
    chi2bbufs.push_back(cl::Buffer(context, CL_MEM_READ_WRITE, 2 * nTracks * sizeof(float)));
  }
  measvarbuf = cl::Buffer(context, CL_MEM_READ_ONLY, 2 * nPlanes * sizeof(float));
  invscatterbuf = cl::Buffer(context, CL_MEM_READ_ONLY, nPlanes * sizeof(float));
  dzbuf = cl::Buffer(context, CL_MEM_READ_ONLY, nPlanes * sizeof(float));
    
  //reducedbuf = cl::Buffer(context, CL_MEM_READ_WRITE, nCudaCores * 2 * sizeof(float));
  for(int ii = 0; ii < nPlanes - 2; ii++){
    reducedxbufs.push_back(cl::Buffer(context, CL_MEM_READ_WRITE, 2 * nCudaCores * sizeof(float)));
    reducedybufs.push_back(cl::Buffer(context, CL_MEM_READ_WRITE, 2 * nCudaCores * sizeof(float)));
  }
  
  //Allocate arrays for data transfer
  measX = (float**) malloc(nPlanes * sizeof(float*));
  for(int ii = 0; ii < nPlanes; ii++){
    measX[ii] = (float*) malloc(2 * nTracks * sizeof(float));
  }
  //This will be read back after reduction
  chi2f = (float**) malloc((nPlanes - 2) * sizeof(float*));
  chi2b = (float**) malloc((nPlanes - 2) * sizeof(float*));
  for(int ii = 0; ii < (nPlanes -2); ii++){
    chi2f[ii] = (float*) malloc(2 * nCudaCores * sizeof(float));
    chi2b[ii] = (float*) malloc(2 * nCudaCores * sizeof(float));
  }
  
  measVar = (float*) malloc(2 * nPlanes * sizeof(float));
  invScatter = (float*) malloc(nPlanes * sizeof(float));
  dz = (float*) malloc(nPlanes * sizeof(float));
}

void SDR2CL::init(){
  //Read measurements into float** array from the MatEst object.
  if(not readTracks){
    mat.readTracksToDoubleArray(measX, nTracks, nPlanes);
    readTracks = true;
    //Write measurements to kernel
    for(int ii = 0; ii < nPlanes; ii++){
      queue.enqueueWriteBuffer(measbufs.at(ii), CL_TRUE, 0, 2 * nTracks * sizeof(float), measX[ii]); 
    }
  }
  Minimizer::init();
}

//Stage two of the two-stage reduction
//Elements alternate between x and y.
void SDR2CL::reduceStage2(cl::Buffer &buf, float* chi2){
  queue.enqueueReadBuffer(buf, CL_TRUE, 0, 2 * nCudaCores * sizeof(float), chi2);
  float sumx = 0.0f;
  float sumy = 0.0f;
  for(int ii = 0 ; ii < (nCudaCores * 2); ii+=2){
    sumx += chi2[ii];
    sumy += chi2[ii+1];
  }
  float resSum = 1.0f - (sumx/nTracks);
  {
#ifdef DOTHREAD
    boost::mutex::scoped_lock lock(reduceGurad);
#endif
    result += resSum * resSum;
    resSum = 1.0f - (sumy/nTracks);
    result += resSum * resSum;
  }
}

void SDR2CL::writeSysParams(){
  //System parameters that will be passed to kernel
  for(int ii = 0; ii < nPlanes; ii++){
    measVar[(ii * 2)]     = (mat.resX.at(ii) * mat.resX.at(ii));
    measVar[(ii * 2) + 1] = (mat.resY.at(ii) * mat.resY.at(ii));
    invScatter[ii] = 1.0f/fabs(mat.system.planes.at(ii).getScatterThetaSqr());
  }
  dz[0] = 0.0f;
  for(int ii = 1; ii < nPlanes; ii++){
    dz[ii] = mat.zPos[ii] - mat.zPos[ii - 1];
  }
  //Copy to kernel
  queue.enqueueWriteBuffer(measvarbuf, CL_TRUE, 0, 2 * nPlanes * sizeof(float), measVar);
  queue.enqueueWriteBuffer(invscatterbuf, CL_TRUE, 0, nPlanes * sizeof(float), invScatter);
  queue.enqueueWriteBuffer(dzbuf, CL_TRUE, 0, nPlanes * sizeof(float), dz);
}

void SDR2CL::operator() (size_t offset, size_t /*stride*/){
  //Fit all tracks, extract the variance of the variance - 1 for pull distributions.
  //Force a single host, discard stride
  if(offset!=0) {return;} 

  writeSysParams();
  
  int arg = 0;
  //Set arguments
  for(int ii = 0; ii < nPlanes; ii++){
    fitPlanes.setArg(arg++, measbufs.at(ii));
  }
  for(int ii = 0; ii < nPlanes - 2; ii++){
    fitPlanes.setArg(arg++, chi2fbufs.at(ii));
  }
  for(int ii = 0; ii < nPlanes - 2; ii++){
    fitPlanes.setArg(arg++, chi2bbufs.at(ii));
  }
  fitPlanes.setArg(arg++, measvarbuf);
  fitPlanes.setArg(arg++, invscatterbuf);
  fitPlanes.setArg(arg++, dzbuf);
  fitPlanes.setArg(arg++, nPlanes * 2 * sizeof(float), NULL);
  fitPlanes.setArg(arg++, nPlanes * sizeof(float), NULL);
  fitPlanes.setArg(arg++, nPlanes * sizeof(float), NULL);

  //Kernel uses float8, one flot4 for x, one float4 for y.
  cl::NDRange global(nTracks/nParallel);
  cl::NDRange local(32);
  queue.enqueueNDRangeKernel(fitPlanes, cl::NullRange, global, local);

  result = 0;
  //Two-stage reduce, first stage on GPU, second on CPU threads
  
  for(int pl = 0; pl < (nPlanes - 2); pl++){
    reduce.setArg(0, chi2fbufs.at(pl));
    reduce.setArg(1, 2 * workGroupSize * sizeof(float), NULL);
    reduce.setArg(2, nTracks);
    reduce.setArg(3, reducedxbufs.at(pl));
    
    cl::NDRange nt(nReduceThreads);
    cl::NDRange wg(workGroupSize);
    queue.enqueueNDRangeKernel(reduce, cl::NullRange, nt, wg);
#ifdef DOTHREAD
    threads.create_thread( boost::bind(&SDR2CL::reduceStage2, this, reducedxbufs.at(pl), chi2f[pl]) );
#else
    reduceStage2(reducedxbufs.at(pl),chi2f[pl]);
#endif
  }
  
  //BW
  for(int pl = 0; pl < (nPlanes - 2); pl++){
    reduce.setArg(0, chi2bbufs.at(pl));
    reduce.setArg(1, 2 * workGroupSize * sizeof(float), NULL);
    reduce.setArg(2, nTracks);
    reduce.setArg(3, reducedybufs.at(pl));
    
    cl::NDRange nt(nReduceThreads);
    cl::NDRange wg(workGroupSize);
    queue.enqueueNDRangeKernel(reduce, cl::NullRange, nt, wg);
    
#ifdef DOTHREAD
    threads.create_thread( boost::bind(&SDR2CL::reduceStage2, this, reducedybufs.at(pl), chi2b[pl]) );
#else
    reduceStage2(reducedybufs.at(pl), chi2b[pl]);
#endif
  }
#ifdef DOTHREAD
  threads.join_all();
#endif
}


//Full read 5:14. Quarter read :

SDR2CL::~SDR2CL(){
  for(int ii = 0; ii < nPlanes; ii++){
    free(measX[ii]);
  }
  free(measX);
  free(measVar);
  free(invScatter);
  free(dz);
  for(int ii = 0; ii < nPlanes - 2; ii++){
    free(chi2f[ii]);
    free(chi2b[ii]);
  }
  free(chi2f);
  free(chi2b);
}
