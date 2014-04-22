#include "estmat.h"
#include "sdr2clt3.h"

void estimationParameters(EstMat& mat){
  //Setting up parameters for minimization. 
  //Pushing a plane index to the vector means the parameter will be estimated
  //Identical plane resolutions
  // mat.resXYMulti.push_back(0);
  // mat.resXYMulti.push_back(1);
  // mat.resXYMulti.push_back(2);
  // mat.resXYMulti.push_back(6);
  // mat.resXYMulti.push_back(7);
  // mat.resXYMulti.push_back(8);
  
  //Individual plane resolutions
  mat.resXIndex.push_back(3);
  mat.resXIndex.push_back(4);
  mat.resXIndex.push_back(5);
  mat.resYIndex.push_back(3);
  mat.resYIndex.push_back(4);
  mat.resYIndex.push_back(5);
  
  //Identical plane thicknesses
  //mat.radLengthsMulti.push_back(0);
  // mat.radLengthsMulti.push_back(1);
  // mat.radLengthsMulti.push_back(2);
  // mat.radLengthsMulti.push_back(6);
  // mat.radLengthsMulti.push_back(7);
  //mat.radLengthsMulti.push_back(8);
  
  //Individual plane thicknesses
  mat.radLengthsIndex.push_back(3);
  mat.radLengthsIndex.push_back(4);
  mat.radLengthsIndex.push_back(5);
}

void simulateTracks(EstMat& mat, int nTracks){
  //Set up the nominal thicknesses and resolution
  //TEL
  for(int ii = 0; ii < mat.system.planes.size(); ii++){
    mat.resX.at(ii) = mat.resY.at(ii) = 4.3;
    mat.radLengths.at(ii) = 0.0073;
    mat.setPlane(ii, mat.resX.at(ii), mat.resY.at(ii), mat.radLengths.at(ii));
  }
  
  //DUT
  mat.radLengths.at(3) = 0.1010;
  mat.radLengths.at(4) = 0.1015;
  mat.radLengths.at(5) = 0.0837;
  for(int ii = 3; ii < 6; ii++){
    mat.resX.at(ii) = 15.0;//14.43;
    mat.resY.at(ii) = 115.0;//115.47;
    mat.setPlane(ii, mat.resX.at(ii), mat.resY.at(ii), mat.radLengths.at(ii));
  }
  mat.clear();
  mat.simulate( nTracks );
  std::cout << "Done simulating" << endl;
}

void initialGuess(EstMat& mat){
  //Set up initial guesses for the minimization

  //TEL, set to true values
  for(int ii = 0; ii < mat.system.planes.size(); ii++){
    mat.resX.at(ii) = mat.resY.at(ii) = 4.3;
    mat.radLengths.at(ii) = 0.0073;
    mat.setPlane(ii, mat.resX.at(ii), mat.resY.at(ii), mat.radLengths.at(ii));
  }
  
  //DUT, gaussian smear from truth
  for(int ii = 3; ii < 6; ii++){
    double gr1(0), gr2(0);
    gaussRand(gr1, gr2);
    mat.resX.at(ii) = 15.0 + 3.0 * gr1;
    mat.resY.at(ii) = 115.0 + 3.0 * gr2;
    gaussRand(gr1, gr2);
    mat.radLengths.at(ii) = 0.08 + 0.02 * gr1;
    mat.setPlane(ii, mat.resX.at(ii), mat.resY.at(ii), mat.radLengths.at(ii));
  }
}

void freeAndSetAlignVec(vector<int>& indexes, vector<FITTERTYPE>& values, double sigma){
  indexes.clear();
  for(int ii = 1; ii < 8; ii++){
    indexes.push_back(ii);
    double a, b;
    gaussRand(a,b);
    values.at(ii) = a * sigma;
  }
}

void prepareAlignment(EstMat& mat){
  mat.printAllFreeParams();

  mat.resXIndex.clear();
  mat.resYIndex.clear();
  mat.radLengthsIndex.clear();

  freeAndSetAlignVec(mat.xShiftIndex, mat.xShift, 500); //Gaussian spread of x shifts, 500um sigma around truth = 0um
  freeAndSetAlignVec(mat.yShiftIndex, mat.yShift, 500); //Gaussian spread of y shifts, 500um sigma around truth = 0um
  freeAndSetAlignVec(mat.xScaleIndex, mat.xScale, 0.1); //Gaussian spread of y scales, 0 is truth
  freeAndSetAlignVec(mat.yScaleIndex, mat.yScale, 0.1); 
  freeAndSetAlignVec(mat.zRotIndex, mat.zRot, 0.1); 
}

int main(int argc, char* argv[]){

  if(argc < 2) {
    cout << "Needs an argument, fwbw, sdr1, sdr2,, sdr2cl, sdr3, hybr or align" << endl;
    return(1);
  }

  //Configure system, simulate tracks, and estimate material and resolution
  double ebeam = 40.0; //Beam energy
  int nPlanes = 9;
  int nTracks = 1000000; //How many tracks to simulate per experiment
  //int nTracks = 5; //How many tracks to simulate per experiment
  int numberOfExperiments = 1; //How many simulation + estimation estimates should be preformed

  EstMat mat;
  mat.init(ebeam, nPlanes); //Initialize the the estimator
  mat.initSim(nPlanes); //Initialize the simulation

  //Z positions of planes
  mat.movePlaneZ(0, 10    ); //Move plane 0 to 10mum
  mat.movePlaneZ(1, 150010);
  mat.movePlaneZ(2, 300010);
  mat.movePlaneZ(3, 361712);
  mat.movePlaneZ(4, 454810);
  mat.movePlaneZ(5, 523312);
  mat.movePlaneZ(6, 680010);
  mat.movePlaneZ(7, 830010);
  mat.movePlaneZ(8, 980010);

  //Initialize histograms for the three APIX planes
  std::vector<TH1D*> xposes;
  std::vector<TH1D*> yposes;
  std::vector<TH1D*> radests;
  for( int ii = 3; ii < 6; ii++) {
    char * name = new char[50];
    sprintf(name, "x%i", ii);
    xposes.push_back(new TH1D(name, name,100, 14.0, 16.0));
    sprintf(name, "y%i", ii);
    yposes.push_back(new TH1D(name, name,100, 110.0, 120.0));
    sprintf(name, "rad%i", ii);
    radests.push_back(new TH1D(name, name,100, 0.06, 0.12));
  }
  
  estimationParameters(mat); //Configure which parameters to free in the fit
  
  //For each outeriter, a track sample will be simulated, and the system parameters will be estimated
  for(int outeriter = 0; outeriter < numberOfExperiments; outeriter ++){
    cout << "Outer iteration " << outeriter << endl;
    
    simulateTracks(mat, nTracks); //Configure true state in this function
    initialGuess(mat); //Set up initial guesses for material and resolutions, 
    //mat.plot( (char*) "test.root");

    //alignment should work with bad resolution and material budget estimates
    if(strcmp(argv[1], "align") == 0){
      prepareAlignment(mat); //Set up initial guesses for alignment parameters
    }
    //Set up the material estimation from the Tracker System
    for(size_t plane = 0; plane < mat.system.planes.size(); plane ++){
      mat.resX.at(plane) = mat.system.planes.at(plane).getSigmas()(0);
      mat.resY.at(plane) = mat.system.planes.at(plane).getSigmas()(1);
      mat.zPos.at(plane) = mat.system.planes.at(plane).getZpos();
      double scatterSigma = getScatterSigma(mat.eBeam, mat.radLengths.at(plane));
      mat.system.planes.at(plane).setScatterThetaSqr( scatterSigma * scatterSigma);
    }

    //Print the plane parameters
    for(size_t plane = 0; plane < mat.system.planes.size(); plane ++){
      mat.system.planes.at(plane).print();
    }

    //Parsing argument, doing minimization
    int iterations = 300; //How many iterations per restart?
    int restarts = 3; // How many times should the simplex search be restarted?

    Minimizer* minimize = NULL;

    if(strcmp(argv[1], "fwbw") == 0){
      cout << "Starting minimization of type fwbw" << endl;
      minimize = new FwBw(mat);
      mat.simplexSearch(minimize, iterations, restarts);
    } else if(strcmp(argv[1], "sdr2cl") == 0){
      cout << "Starting minimization of type SDR2CL" << endl;
      if(numberOfExperiments > 1){
	cout << "SDR2CL has some problems with releasing openCL resources. The program might get slow and crash if many jobs are run." << endl;
      }
      minimize = new SDR2CL(mat, nPlanes, nTracks);
      mat.simplexSearch(minimize, iterations, restarts);
    } else if(strcmp(argv[1], "sdr1") == 0){
      cout << "Starting minimization of type SDR1" << endl;
      minimize = new SDR(true,false,false,mat);
      mat.simplexSearch(minimize, iterations, restarts);
    } else if(strcmp(argv[1], "sdr1c") == 0){
      cout << "Starting minimization of type SDR1c" << endl;
      minimize = new SDR(true,false,true,mat);
      mat.simplexSearch(minimize, iterations, restarts);
    } else if(strcmp(argv[1], "sdr2") == 0){
      cout << "Starting minimization of type SDR2" << endl;
      minimize = new SDR(false,true,false,mat);
      mat.simplexSearch(minimize, iterations, restarts);
    } else if(strcmp(argv[1], "sdr3") == 0){
      cout << "Starting minimization of type SDR3" << endl;
      minimize = new SDR(true,true,false,mat);
      mat.simplexSearch(minimize, iterations, restarts);
    } else if(strcmp(argv[1], "fake") == 0){
      cout << "Starting minimization of type fake chi2" << endl;
      minimize = new FakeChi2(mat);
      mat.simplexSearch(minimize, iterations, restarts);
    } else if(strcmp(argv[1], "hybr") == 0){
      cout << "Starting minimization of type hybr" << endl;
      minimize = new FwBw(mat);
      mat.quasiNewtonHomeMade(minimize, 40);
    } else if(strcmp(argv[1], "align") == 0){
      cout << "Running alignment!" << endl;
      minimize = new FakeChi2(mat);
      //Minimizer* minimizer = new FwBw(mat); //<- Should also work, but slower and seems to preform slightly worse (no real comparison performed)
      //Simplex search, starting with 1k tracks, then full sample
      cout << "1K tracks" << endl;
      mat.simplexSearch(minimize, 2000, 15, 1000);
      cout << "Full track sample" << endl;
      mat.simplexSearch(minimize, 3000, 1, nTracks);
      //mat.quasiNewtonHomeMade(minimize, 100);
      return(0); //No plots are filled, no reason to do it 100 times
    } else {
      cout << "Needs an argument, fwbw, sdr1, sdr2, sdr2cl, sdr3, hybr or align" << endl;
      return(1);
    }

    if(minimize != NULL){
      delete minimize;
    }

    cout << "Done estimating" << endl << endl << endl;
    
    //Plot pull distributions, residuals and chi squares
    char* plotname = new char[200];
    sprintf(plotname, "plots/iteration%i%s.root", (int)outeriter, argv[1]); 
    //mat.plot(plotname);
    delete [] plotname;
    
    //Fill histograms of estimates and plot
    for( int ii = 3; ii < 6; ii++) {
      xposes.at( ii - 3)->Fill( fabs(mat.resX.at(ii)));
      yposes.at( ii - 3)->Fill( fabs(mat.resY.at(ii)));
      radests.at( ii - 3)->Fill( fabs(mat.radLengths.at(ii)));
    }
    char name[100];
    sprintf(name,"plots/boot%s-%i.root", argv[1], int(mat.eBeam));
    TFile* tfile = new TFile(name, "RECREATE");
    
    for( int ii = 0; ii < 3; ii++) {
      xposes.at( ii )->Write();
      yposes.at( ii )->Write();
      radests.at( ii )->Write();
    }
    tfile->Close();
    delete tfile;
  }
}
