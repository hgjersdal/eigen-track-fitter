#include <iostream>
#include <iomanip>
#include <cmath>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <vector>
#include <map>
#include <TFile.h>
#include <TH1D.h>
#include <TMath.h>

#include <Eigen/Core>
#include <Eigen/Array>
#include <Eigen/LU>
#include <Eigen/Cholesky>

#include "EUTelDafTrackerSystem.h"


#include <gsl/gsl_vector_double.h>

using namespace daffitter;
//typedef double FITTERTYPE;
typedef float FITTERTYPE;

class Minimizer;
class FwBw;

class EstMat{
public:
  //parameters
  std::vector<FITTERTYPE> resX, resY, radLengths, zPos, xShift, yShift, xScale, yScale, zRot;
  //single parameter iteration indexes
  std::vector<int> resXIndex, resYIndex, resXYIndex, radLengthsIndex, zPosIndex, xShiftIndex, yShiftIndex, xScaleIndex, yScaleIndex, zRotIndex;
  //multi parameter iteration indexes
  std::vector<int> resXMulti, resYMulti, resXYMulti, radLengthsMulti;
  //ebeam
  double eBeam;
  //data
  std::vector< std::vector<Measurement<FITTERTYPE> > > tracks;
  TrackerSystem<FITTERTYPE, 4> system;

  //constructor
  void init(double eBeam, size_t nPlanes){
    this->eBeam = eBeam;
    radLengths.assign(nPlanes, 0.01);
    resX.assign(nPlanes, 4.3);
    resY.assign(nPlanes, 4.3);
    xShift.assign(nPlanes, 0.0);
    yShift.assign(nPlanes, 0.0);
    xScale.assign(nPlanes, 0.0);
    yScale.assign(nPlanes, 0.0);
    zRot.assign(nPlanes, 0.0);
    zPos.assign(nPlanes, 0.0);
  }

  //Action
  void estimate();
  void plot(char* fname);

  //simulation
  void initSim(int nplanes);
  void simulate(int nTracks);

  //initialization
  void setPlane(int index, double sigmaX, double sigmaY, double radLength);
  void addPlane(FitPlane<FITTERTYPE>& pl);
  void addTrack( std::vector<Measurement<FITTERTYPE> > track);

  void movePlaneZ(int planeIndex, double deltaZ);
  double minimizeMe();

  size_t getNSimplexParams();
  void estToSystem( const gsl_vector* params);
  gsl_vector* systemToEst();
  gsl_vector* simplesStepSize();
  void simplexSearch(Minimizer* minimizeMe);
  void quasiNewtonHomeMade(FwBw* minimizeMe);
  
  int itMax;
  void readTrack(int track);
  void clear(){ tracks.clear(); }
  void getExplicitEstimate(TrackEstimate<FITTERTYPE, 4>* estim);
  void parameterIteration( double step, double Orig, double min, double max, int index, void (*stepFun) (EstMat& mat, int index, double step)  );
  void printParams( char* name, std::vector<FITTERTYPE>& params, bool plot);
};

class Minimizer{
public:
  EstMat& mat;
  FITTERTYPE retVal2;
  Minimizer(EstMat& mat) : mat(mat) {;}
  virtual FITTERTYPE operator() (void) = 0;
};

class Chi2: public Minimizer {
public:
  Chi2(EstMat& mat) : Minimizer(mat) {;}
  virtual FITTERTYPE operator()(void);
};

class SDR: public Minimizer {
private:
  bool SDR1, SDR2, cholDec;
public:
  SDR(bool SDR1, bool SDR2, bool cholDec,  EstMat& mat): Minimizer(mat), SDR1(SDR1), SDR2(SDR2), cholDec(cholDec) {;}
  virtual FITTERTYPE operator()(void);
};

class FwBw: public Minimizer {
public:
  FITTERTYPE retVal2;
  FwBw(EstMat& mat): Minimizer(mat) {;}
  virtual FITTERTYPE operator()(void);
};

double normRand();
void gaussRand(double& x1, double& x2);
double getScatterSigma(double eBeam, double radLength);

