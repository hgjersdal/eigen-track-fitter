#include <utility>
#include <iostream>
#include <fstream>
#include <string>
#include "estmat.h"
#include "sdr2clt3.h"

SDR2CL::SDR2CL(EstMat& mat, int nplanes, int ntracks) : Minimizer(mat), nPlanes(nplanes), readTracks(false), nTracks(ntracks),
							nGPUThreads(nTracks/4) {
  // Get available platforms
  vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);
  
  // Select the default platform and create a context using this platform and the GPU
  cl_context_properties cps[3] = { 
    CL_CONTEXT_PLATFORM, 
    (cl_context_properties)(platforms[0])(), 
    0 
  };
  context = cl::Context( CL_DEVICE_TYPE_GPU, cps);
  
  // Get a list of devices on this platform
  vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
  
  // Create a command queue and use the first device
  queue = cl::CommandQueue(context, devices[0]);
  
  cout << "Device: " << endl;
  cout << devices[0].getInfo<CL_DEVICE_NAME>() << endl;

  // Read source file
  std::ifstream sourceFile("src/openclkalmant3.cl");
  std::string sourceCode( std::istreambuf_iterator<char>(sourceFile),
			 (std::istreambuf_iterator<char>()));
  cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length()+1));
  
  // Make program of the source code in the context
  cl::Program program = cl::Program(context, source);
  
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
  firstFW = cl::Kernel(program, "processFirstPlaneFW");
  secondFW = cl::Kernel(program, "processSecondPlaneFW");
  restFW = cl::Kernel(program, "processNormalPlaneFW");
  
  firstBW = cl::Kernel(program, "processFirstPlaneBW");
  secondBW = cl::Kernel(program, "processSecondPlaneBW");
  restBW = cl::Kernel(program, "processNormalPlaneBW");
  
  // Create memory buffers 
  //read-write
  bufx = cl::Buffer(context, CL_MEM_READ_WRITE, nTracks * sizeof(float));
  bufdx = cl::Buffer(context, CL_MEM_READ_WRITE, nTracks * sizeof(float));
  bufxx = cl::Buffer(context, CL_MEM_READ_WRITE, nTracks * sizeof(float));
  bufxdx = cl::Buffer(context, CL_MEM_READ_WRITE, nTracks * sizeof(float));
  bufdxdx = cl::Buffer(context, CL_MEM_READ_WRITE, nTracks * sizeof(float));
  //write-only
  bufchi2x = cl::Buffer(context, CL_MEM_WRITE_ONLY, nGPUThreads * sizeof(float));
  //read-only
  bufmeasX = cl::Buffer(context, CL_MEM_READ_ONLY, nTracks * sizeof(float));
  //read-write
  bufy = cl::Buffer(context, CL_MEM_READ_WRITE, nTracks * sizeof(float));
  bufdy = cl::Buffer(context, CL_MEM_READ_WRITE, nTracks * sizeof(float));
  bufyy = cl::Buffer(context, CL_MEM_READ_WRITE, nTracks * sizeof(float));
  bufydy = cl::Buffer(context, CL_MEM_READ_WRITE, nTracks * sizeof(float));
  bufdydy = cl::Buffer(context, CL_MEM_READ_WRITE, nTracks * sizeof(float));
  //write-only
  bufchi2y = cl::Buffer(context, CL_MEM_WRITE_ONLY, nGPUThreads * sizeof(float));
  //read-only
  bufmeasY = cl::Buffer(context, CL_MEM_READ_ONLY, nTracks * sizeof(float));

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
  if(not readTracks){
    cout << "Reading measurements!" << endl;
    mat.readTracksToArray(measX, measY, nTracks, nPlanes);
    readTracks = true;
  }
  Minimizer::init();
}

void SDR2CL::copyMeasurements(float* measx, float* measy){
  queue.enqueueWriteBuffer(bufmeasX, CL_TRUE, 0, nTracks * sizeof(float), measx); 
  queue.enqueueWriteBuffer(bufmeasY, CL_TRUE, 0, nTracks * sizeof(float), measy); 
}

void SDR2CL::runKernel(cl::Kernel& kernel, float resx, float resy, float scattervar, float dz){
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
  queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local);
}

//Threaded reduce
void SDR2CL::startReduction(int nThreads){
  //Reset counters
  resVarX = 0.0f;
  resVarY = 0.0f;
  //Read data
  queue.enqueueReadBuffer(bufchi2x, CL_TRUE, 0, nGPUThreads * sizeof(float), chi2x);
  queue.enqueueReadBuffer(bufchi2y, CL_TRUE, 0, nGPUThreads * sizeof(float), chi2y);
  //Start threads
#ifdef DOTHREAD
  int nElements = nGPUThreads/nThreads; 
  for(int ii = 0; ii < nThreads; ii++){
    threads.create_thread( boost::bind(&SDR2CL::threadReduce, this, ii * nElements, (ii + 1) * nElements));
    //threads.create_thread( boost::bind(&SDR2CL::threadReduce, this, ii, nThreads));
  }
#else
  threadReduce(0,1);
#endif    
}

//geforce gtx 460

void SDR2CL::threadTally(){
  //Join threads, tally result
#ifdef DOTHREAD
  threads.join_all();
#endif
  //Tally
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
  if(offset!=0) {return;} //Force a single host, discard stride
  //Forward fit
  bool waitReduce = false;
  int numThreads = 1;
  
  copyMeasurements(measX[0], measY[0]);
  runKernel(firstFW, mat.resX.at(0), mat.resY.at(0),
	    mat.system.planes.at(0).getScatterThetaSqr(),
	    0.0f);
  copyMeasurements(measX[1], measY[1]);
  runKernel(secondFW, mat.resX.at(1), mat.resY.at(1),
	    mat.system.planes.at(1).getScatterThetaSqr(),
	    mat.zPos[1] - mat.zPos[0]);
  for(int pl = 2; pl < nPlanes; pl++){
    copyMeasurements(measX[pl], measY[pl]);
    runKernel(restFW, mat.resX.at(pl), mat.resY.at(pl),
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

    // queue.enqueueReadBuffer(bufchi2x, CL_TRUE, 0, nTracks * sizeof(float), chi2x);
    // queue.enqueueReadBuffer(bufchi2y, CL_TRUE, 0, nTracks * sizeof(float), chi2y);
    // result += chi2x[nTracks - 1];
    // result += chi2y[nTracks - 1];
    // result += chi2x[0];
    // result += chi2y[0];
  }
  
  int pl = nPlanes - 1;
  copyMeasurements(measX[pl], measY[pl]);
  runKernel(firstBW, mat.resX.at(pl), mat.resY.at(pl),
	    mat.system.planes.at(pl).getScatterThetaSqr(),
	    0.0f);

  pl = nPlanes - 2;
  copyMeasurements(measX[pl], measY[pl]);
  runKernel(secondBW, mat.resX.at(pl), mat.resY.at(pl),
	    mat.system.planes.at(pl).getScatterThetaSqr(),
	    mat.zPos[pl] - mat.zPos[pl+1]);

  for(int pl = nPlanes -3; pl >= 0; pl--){
    copyMeasurements(measX[pl], measY[pl]);
    runKernel(restBW, mat.resX.at(pl), mat.resY.at(pl),
	      mat.system.planes.at(pl).getScatterThetaSqr(),
	      mat.zPos[pl] - mat.zPos[pl+1]);
    //Process chi2s
    threadTally();
    startReduction(numThreads);
    // queue.enqueueReadBuffer(bufchi2x, CL_TRUE, 0, nTracks * sizeof(float), chi2x);
    // queue.enqueueReadBuffer(bufchi2y, CL_TRUE, 0, nTracks * sizeof(float), chi2y);
    // result += chi2x[nTracks - 1];
    // result += chi2y[nTracks - 1];
    // result += chi2x[0];
    // result += chi2y[0];
  }
  threadTally();
}

SDR2CL::~SDR2CL(){
  for(int ii = 0; ii < nPlanes; ii++){
    free(measX[ii]);
    free(measY[ii]);
  }
  free(measX);
  free(measY);
  free(chi2x);
  free(chi2y);
}
