#include <utility>
#include <iostream>
#include <fstream>
#include <string>
#include <unistd.h>
#include "estmat.h"
#include "sdr2clt3.h"

//For some reason, the destructor does not release all resources, and the 31th time the
//constructor is called, an error of type CL_DEVICE_NOT_AVAILABLE occurs. For this reason,
//the openCL resources are made global. This also avoids recompilation of kernels.
namespace SDR2CL_glob {
  bool initialized = false;
  std::vector<cl::Platform> platforms;
  cl::Context context;
  cl::CommandQueue queue;
  
  cl::Program program;
  cl::Kernel firstFW, secondFW, restFW;
  cl::Kernel firstBW, secondBW, restBW;
}

//Constructor
SDR2CL::SDR2CL(EstMat& mat, int nplanes, int ntracks) : Minimizer(mat), nPlanes(nplanes), readTracks(false), nTracks(ntracks),
							nGPUThreads(nTracks/8) {
  if(nGPUThreads * 8 < nTracks){
    cout << nTracks - nGPUThreads * 8 << " tracks will not be analyzed." << endl;
  }
  //Initialize CL and compile kernels once.
  if(not SDR2CL_glob::initialized){
    SDR2CL_glob::initialized = true;
    cout << "Initializing kernels!" << endl;
    // Get available platforms
    cl::Platform::get(&SDR2CL_glob::platforms);
    
    // Select the default platform and create a context using this platform and the GPU
    cl_context_properties cps[3] = { 
      CL_CONTEXT_PLATFORM, 
      (cl_context_properties)(SDR2CL_glob::platforms[0])(), 
      0 
    };
    try{
      SDR2CL_glob::context = cl::Context( CL_DEVICE_TYPE_GPU, cps);
    }  catch(cl::Error error) {
      std::cout << error.what() << "(" << error.err() << ")" << std::endl;
    }
    // Get a list of devices on this platform
    vector<cl::Device> devices = SDR2CL_glob::context.getInfo<CL_CONTEXT_DEVICES>();
    
    // Create a command queue and use the first device
    SDR2CL_glob::queue = cl::CommandQueue(SDR2CL_glob::context, devices[0]);
    
    cout << "Device: " << endl;
    cout << devices[0].getInfo<CL_DEVICE_NAME>() << endl;
    
    // Read source file
    std::ifstream sourceFile("src/openclkalmant5.cl");
    std::string sourceCode( std::istreambuf_iterator<char>(sourceFile),
			    (std::istreambuf_iterator<char>()));
    cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length()+1));
    
    // Make program of the source code in the context
    cl::Program program = cl::Program(SDR2CL_glob::context, source);
    
    // Build program for these specific devices
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
    SDR2CL_glob::firstFW = cl::Kernel(program, "processFirstPlaneFW");
    SDR2CL_glob::secondFW = cl::Kernel(program, "processSecondPlaneFW");
    SDR2CL_glob::restFW = cl::Kernel(program, "processNormalPlaneFW");
    
    SDR2CL_glob::firstBW = cl::Kernel(program, "processFirstPlaneBW");
    SDR2CL_glob::secondBW = cl::Kernel(program, "processSecondPlaneBW");
    SDR2CL_glob::restBW = cl::Kernel(program, "processNormalPlaneBW");
  }//End initialization of global CL kernels

  // Create memory buffers 
  //read-write
  bufx = cl::Buffer(SDR2CL_glob::context, CL_MEM_READ_WRITE, nTracks * sizeof(float));
  bufdx = cl::Buffer(SDR2CL_glob::context, CL_MEM_READ_WRITE, nTracks * sizeof(float));
  bufxx = cl::Buffer(SDR2CL_glob::context, CL_MEM_READ_WRITE, nTracks * sizeof(float));
  bufxdx = cl::Buffer(SDR2CL_glob::context, CL_MEM_READ_WRITE, nTracks * sizeof(float));
  bufdxdx = cl::Buffer(SDR2CL_glob::context, CL_MEM_READ_WRITE, nTracks * sizeof(float));

  bufy = cl::Buffer(SDR2CL_glob::context, CL_MEM_READ_WRITE, nTracks * sizeof(float));
  bufdy = cl::Buffer(SDR2CL_glob::context, CL_MEM_READ_WRITE, nTracks * sizeof(float));
  bufyy = cl::Buffer(SDR2CL_glob::context, CL_MEM_READ_WRITE, nTracks * sizeof(float));
  bufydy = cl::Buffer(SDR2CL_glob::context, CL_MEM_READ_WRITE, nTracks * sizeof(float));
  bufdydy = cl::Buffer(SDR2CL_glob::context, CL_MEM_READ_WRITE, nTracks * sizeof(float));
  //write-only
  bufchi2x = cl::Buffer(SDR2CL_glob::context, CL_MEM_WRITE_ONLY, nGPUThreads * sizeof(float));
  bufchi2y = cl::Buffer(SDR2CL_glob::context, CL_MEM_WRITE_ONLY, nGPUThreads * sizeof(float));

  //read-only
  bufmeasX = cl::Buffer(SDR2CL_glob::context, CL_MEM_READ_ONLY, nTracks * sizeof(float));
  bufmeasY = cl::Buffer(SDR2CL_glob::context, CL_MEM_READ_ONLY, nTracks * sizeof(float));

  //Allocate arrays for data transfer
  measX = (float**) malloc(nPlanes * sizeof(float*));
  measY = (float**) malloc(nPlanes * sizeof(float*));
  for(int ii = 0; ii < nPlanes; ii++){
    measX[ii] = (float*) malloc(nTracks * sizeof(float));
    measY[ii] = (float*) malloc(nTracks * sizeof(float));
  }
  chi2x= (float*) malloc(nGPUThreads * sizeof(float));
  chi2y= (float*) malloc(nGPUThreads * sizeof(float));
}

void SDR2CL::init(){
//Read measurements from into float** arrays from the MatEst object..
  if(not readTracks){
    cout << "Reading measurements!" << endl;
    mat.readTracksToArray(measX, measY, nTracks, nPlanes);
    readTracks = true;
  }
  Minimizer::init();
}

void SDR2CL::copyMeasurements(float* measx, float* measy){
//Copy measurements to GPU
  SDR2CL_glob::queue.enqueueWriteBuffer(bufmeasX, CL_TRUE, 0, nTracks * sizeof(float), measx); 
  SDR2CL_glob::queue.enqueueWriteBuffer(bufmeasY, CL_TRUE, 0, nTracks * sizeof(float), measy); 
}

void SDR2CL::runKernel(cl::Kernel& kernel, float resx, float resy, float scattervar, float dz){
//Run a kernel.
//All kernels take the same arguments.
  kernel.setArg(0, bufmeasX);
  kernel.setArg(1, bufmeasY);
  kernel.setArg(2, bufx);
  kernel.setArg(3, bufdx);
  kernel.setArg(4, bufxx);
  kernel.setArg(5, bufxdx);
  kernel.setArg(6, bufdxdx);
  kernel.setArg(7, bufy);
  kernel.setArg(8, bufdy);
  kernel.setArg(9, bufyy);
  kernel.setArg(10, bufydy);
  kernel.setArg(11, bufdydy);
  kernel.setArg(12, bufchi2x);
  kernel.setArg(13, bufchi2y);
  kernel.setArg(14, resx * resx);
  kernel.setArg(15, resy * resy);
  kernel.setArg(16, 1.0f/scattervar);
  kernel.setArg(17, dz);

  cl::NDRange global(nGPUThreads);
  cl::NDRange local(1);
  SDR2CL_glob::queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local);
}

//Threaded reduce
void SDR2CL::startReduction(int nThreads){
  //Reset counters
  resVarX = 0.0f;
  resVarY = 0.0f;
  //Read data from GPU. 
  SDR2CL_glob::queue.enqueueReadBuffer(bufchi2x, CL_TRUE, 0, nGPUThreads * sizeof(float), chi2x);
  SDR2CL_glob::queue.enqueueReadBuffer(bufchi2y, CL_TRUE, 0, nGPUThreads * sizeof(float), chi2y);
  //Start threads
#ifdef DOTHREAD
  int nElements = nGPUThreads/nThreads; 
  for(int ii = 0; ii < nThreads; ii++){
    threads.create_thread( boost::bind(&SDR2CL::threadReduce, this, ii * nElements, (ii + 1) * nElements));
  }
#else
  threadReduce(0,1);
#endif    
}

void SDR2CL::threadTally(){
  //Join threads, tally result
#ifdef DOTHREAD
  threads.join_all();
#endif
  //Tally results
  float rsx = 1.0f - resVarX /nGPUThreads;
  float rsy = 1.0f - resVarY /nGPUThreads;
  result += rsx * rsx;
  result += rsy * rsy;
}

void SDR2CL::threadReduce(int min, int max){
  //Threaded sum reduce of chi2 vectors
  if( abs(max - nThreads) < 10) { max = nGPUThreads;}
  float xSum(0.0f), ySum(0.0f);
  for(int ii = min; ii < max; ii++){
    xSum += chi2x[ii];
    ySum += chi2y[ii];
  }
  {
#ifdef DOTHREAD
    boost::mutex::scoped_lock lock(reduceGurad);
#endif
    resVarX += xSum;
    resVarY += ySum;
  }
}

void SDR2CL::operator() (size_t offset, size_t /*stride*/){
  //Fit all tracks, extract the variance of the variance - 1 for pull distributions.
  //Force a single host, discard stride
  if(offset!=0) {return;} 
  //Forward fit
  bool waitReduce = false;
  int numThreads = 1;
  
  copyMeasurements(measX[0], measY[0]);
  runKernel(SDR2CL_glob::firstFW, mat.resX.at(0), mat.resY.at(0),
	    mat.system.planes.at(0).getScatterThetaSqr(),
	    0.0f);
  copyMeasurements(measX[1], measY[1]);
  runKernel(SDR2CL_glob::secondFW, mat.resX.at(1), mat.resY.at(1),
	    mat.system.planes.at(1).getScatterThetaSqr(),
	    mat.zPos[1] - mat.zPos[0]);
  for(int pl = 2; pl < nPlanes; pl++){
    copyMeasurements(measX[pl], measY[pl]);
    runKernel(SDR2CL_glob::restFW, mat.resX.at(pl), mat.resY.at(pl),
	      mat.system.planes.at(pl).getScatterThetaSqr(),
	      mat.zPos[pl] - mat.zPos[pl -1]);
    //Process chi2s
    if(waitReduce){
      //Threads started the previous iteration are joned here
      threadTally();
    }
    waitReduce = true;
    //Threads are started here
    startReduction(numThreads);
  }

  //Run the Kalman filter in the beackward direction.
  int pl = nPlanes - 1;
  copyMeasurements(measX[pl], measY[pl]);
  runKernel(SDR2CL_glob::firstBW, mat.resX.at(pl), mat.resY.at(pl),
	    mat.system.planes.at(pl).getScatterThetaSqr(),
	    0.0f);

  pl = nPlanes - 2;
  copyMeasurements(measX[pl], measY[pl]);
  runKernel(SDR2CL_glob::secondBW, mat.resX.at(pl), mat.resY.at(pl),
	    mat.system.planes.at(pl).getScatterThetaSqr(),
	    mat.zPos[pl] - mat.zPos[pl+1]);

  for(int pl = nPlanes -3; pl >= 0; pl--){
    copyMeasurements(measX[pl], measY[pl]);
    runKernel(SDR2CL_glob::restBW, mat.resX.at(pl), mat.resY.at(pl),
	      mat.system.planes.at(pl).getScatterThetaSqr(),
	      mat.zPos[pl] - mat.zPos[pl+1]);
    //Process chi2s
    threadTally();
    startReduction(numThreads);
  }
  threadTally();
}

SDR2CL::~SDR2CL(){
//Destructor
  for(int ii = 0; ii < nPlanes; ii++){
    free(measX[ii]);
    free(measY[ii]);
  }
  free(measX);
  free(measY);
  free(chi2x);
  free(chi2y);

}
