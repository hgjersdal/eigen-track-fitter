#include <cmath>
#include <TH1D.h>
#include <TH2D.h>
#include <TFile.h>
#include <map>
#include <TMath.h>
#include <algorithm>

#include <boost/thread.hpp>
#include <boost/thread/detail/thread_group.hpp>
#include <boost/bind.hpp>

boost::mutex plotGuard;
boost::mutex randGuard;

#include "EUTelDafTrackerSystem.h"

#include "simutils.h"

//#define DAF //CKF + DAF
//#define CLU // CLU + DAF
//no define is CKF + KF

#ifdef CLU //CLU only works with DAF
#define DAF
#endif

/*
Simnple application og the straight line track fitter.

Results are printed to lisp code
*/


//Enum for counters
enum valueBase{
  nMeas = 0,
  nMissed = 1,
  nEvents = 2,
  nAccepted = 3,
  nSuccess = 4,
  nGhost = 5,
  nAdditionalGhosts = 15,
  totWeight = 6,
  realWeight = 7,
  fakeWeight = 8,
  nNan = 9,
};


//Class for bookkeeping
class Results{
  map<int,int> counterMap; //Keep track of simple counters.
  map<int,double> valueMap; //Keep track of values that are not simple counters. i.e. weights.

  Matrix<double, 4, 4> tmpCov; //Used to obtain determinant of empirical covariance of the parameter residuals

public:
  vector< TrackEstimate<float,4> > truth; //The true state of the track, one per plane.

  Results(){ 
    //Initialize counters.
    valueMap[totWeight] = 0.0f;
    valueMap[realWeight] = 0.0f;
    valueMap[fakeWeight] = 0.0f;
    counterMap[nMeas] = 0;
    counterMap[nMissed] = 0;
    counterMap[nEvents] = 0;
    counterMap[nAccepted] = 0;
    counterMap[nSuccess] = 0;
    counterMap[nGhost] = 0;
    counterMap[nAdditionalGhosts] = 0;
    counterMap[nNan] = 0;

    //Prepare for covariance calculation
    tmpCov.setZero();

    //Prepare for 9 planes
    truth.resize(9);
  }

  void incrementCount(int iden){ 
    //Increment a counter
    counterMap[iden]++;
  }
  void incrementValue(int iden, double value){ 
    //Increment a value counter
    valueMap[iden] += value;
  }
  int getCount(int iden){ 
    //Get a count
    return(counterMap[iden]);
  }
  double getValue(int iden){ 
    //Get a value count
    return(valueMap[iden]);
  }
  double getValuePercentage(int numerator, int denominator){ 
    //Get a percentage, avoid divide by 0. Divide by 0 returns 0.
    double val = (getValue(denominator) == 0)? 0 : 100 * getValue(numerator)/ getValue(denominator);
    return(val);
  }
  double getCounterPercentage(int numerator, int denominator){ 
    //Get a percentage, avoid divide by 0. Divide by 0 returns 0.
    double val = (getCount(denominator) == 0)? 0 : 100 * getCount(numerator)/ double(getCount(denominator));
    return(val);
  }
  void increment(Results& tmp){
    //Increment the master result with the thread result
    //Mutex to avoid collisions
    boost::mutex::scoped_lock lock(plotGuard);
    //Counters
    map<int,int>::iterator it1 = tmp.counterMap.begin();
    for(;it1 != tmp.counterMap.end(); it1++){
      counterMap[it1->first] += it1->second;
    }
    //Value counters
    map<int,double>::iterator it2 = tmp.valueMap.begin();
    for(;it2 != tmp.valueMap.end(); it2++){
      valueMap[it2->first] += it2->second;
    }
    //Covariance
    tmpCov += tmp.tmpCov;
  }
  void updateCovariance(Matrix<float, 4, 1>& params, int pl){
    //Increment the covariance matrix. Assumes that the expectation value is 0 for all
    //parameters.
    Matrix<double, 4, 1> difference = params.cast<double>() - truth.at(pl).params.cast<double>();
    for(int ii = 0; ii < 4; ii++){
      for(int jj = 0; jj < 4; jj++){
	tmpCov(ii,jj) += difference(ii) * difference(jj);
      }
    }
  }
  double getGeneralizedVariance(){
    //Get the generalized variance. 
    //Get the covariance matrix
    tmpCov.array() /= double(getCount(nSuccess));
    //Return the determinant
    if( getCount(nSuccess) == 0 ){
      0.0;
    } else {
      return(tmpCov.determinant());
    }
  }
  void printPlist(int noise, const char* hashname){
    //Print results to lisp
    // Insert a propertylist into a hashtable.
    cout << "(setf (gethash " << noise << " " << hashname << ")" << endl << "(list" << endl
	 << " :nmeas "      << getCount(nMeas) << endl
	 << " :nmissed "    << getCount(nMissed) << endl
	 << " :nevents "    << getCount(nEvents) << endl
	 << " :naccepted "  << getCount(nAccepted) << endl
	 << " :nsuccess "   << getCount(nSuccess) << endl
	 << " :nghost "     << getCount(nGhost) << endl
	 << " :nghostadditional "     << getCount(nAdditionalGhosts) << endl
	 << " :nnan "       << getCount(nNan) << endl
	 << " :totweight "  << getValue(totWeight) << endl
	 << " :realweight " << getValue(realWeight) << endl
	 << " :fakeweight " << getValue(fakeWeight) <<  endl
	 << " :purity "     << getValuePercentage(realWeight, totWeight) << endl
	 << " :missing "    << getCounterPercentage(nMissed, nMeas) << endl
	 << " :efficiency " << getCounterPercentage(nSuccess, nEvents)  << endl
	 << " :genvar " << getGeneralizedVariance()  << endl
	 << "))" <<endl;
  }
};


#include "simfunction.h" 

// Function that analyses an event in simulated data.
void analyze(TrackerSystem<float,4>& system, std::vector< std::vector<Measurement<float> > >& simTracks, 
	     Results &result){
  // Run track finder
#ifdef CLU
  system.clusterTracker();
#else
  system.combinatorialKF();
#endif
  
  int myTrack = -1; //-1 means no track is found.
  double chi2ndof = 1000; 
  double ndofmin = 3.5; //Only look at tracks with 4 or more measurements
  for(size_t ii = 0; ii < system.getNtracks(); ii++ ){
    TrackCandidate<float, 4>* track =  system.tracks.at(ii);
    // Fit the track!
#ifdef DAF
    system.fitPlanesInfoDaf( track );
    system.weightToIndex( track );
#else
    system.fitPlanesInfoUnBiased( track );
    system.indexToWeight( track );
#endif
    //Extract the index for the minimum chi2/ndof track
    if( ((track->chi2 / track->ndof) < chi2ndof)
      and (track->ndof > ndofmin)){
    chi2ndof = track->chi2 / track->ndof;
    myTrack = ii;
    }
  }
  
  result.incrementCount(nEvents);
  if(myTrack == -1){ return;  } // -1 means no track was found
  TrackCandidate<float, 4>* track =  system.tracks.at(myTrack);
  //Make a cut in the number of measurements included, and  chi2/ndof
  if(track->ndof < ndofmin){ return; } 
  if( (track->chi2 / track->ndof) > system.getChi2OverNdofCut()) { return; }
  if( isnan(track->chi2 / track->ndof) ){
    cout << "NAN CHI!" << endl; // Should never happen. If it does, a bug in the weigh calculation is a probable cause
    exit(2);
  }
  
  result.incrementCount(nAccepted);
  
  bool ghost = true;
  bool nanp = false;

  float tmptotweight = 0.0f;
  float tmpfakeweight = 0.0f;
  
  for(int ii = 0; ii < track->indexes.size(); ii++ ){
    //Skip planes excluded by model or 
    if(system.planes.at(ii).isExcluded() ) { continue;}
    if(track->indexes.at(ii) < 0 ) {continue;}
    for(int mm = 0; mm < system.planes.at(ii).meas.size(); mm++){
      bool real = system.planes.at(ii).meas.at(mm).goodRegion();
      bool included = ( mm == track->indexes.at(ii) );
#ifdef DAF
      float weight = track->weights.at(ii)(mm);
#else 
      float weight = 0.0f;
      if( included ){ weight = 1.0f;}
#endif
      tmptotweight += weight;
      if(not real){ tmpfakeweight += weight; }
    }
    if( system.planes.at(ii).meas.at(track->indexes.at(ii) ).goodRegion()){
      ghost = false;
    }
    if( (track->weights.at(ii).size() > 0) and isnan(track->weights.at(ii).sum()) ){
      nanp = true;
    }
  }
  if( nanp ){
    result.incrementCount(nNan);
    return;
  }
  if(ghost){
    result.incrementCount(nGhost);
    return;
  }
  if( tmpfakeweight > tmptotweight * 0.5){
    result.incrementCount(nGhost);
    return;
  }
  //Now we have a good track, that is not a ghost track
  result.incrementCount(nSuccess);
  result.updateCovariance(track->estimates.at(3)->params, 3);

  float planeWeight =0.0f;
  float planeFakeWeight = 0.0f;
  for(int pl = 0; pl < system.planes.size(); pl++){
    if(system.planes.at(pl).isExcluded()) {continue;}
    if( track->weights.at(pl).sum() > 1.1){
      cout << "Larger than 1 weights!!! " << track->weights.at(pl).sum() << endl;
    }
    if( track->weights.at(pl).sum() < -0.1){
      cout << "Smaller than 0 weights!!! " << track->weights.at(pl).sum() << endl;
    }
    for(int mm = 0; mm < system.planes.at(pl).meas.size(); mm++){
      bool real = system.planes.at(pl).meas.at(mm).goodRegion();
      bool included = ( mm == track->indexes.at(pl) );
#ifdef DAF
      float weight = track->weights.at(pl)(mm);
#else 
      float weight = 0.0f;
      if( included ){ weight = 1.0f;}
#endif
      if( isnan(weight)){
	cout << weight << ", " << pl << ", " << mm << ", " << result.getCount(nEvents) << endl;;
	cout << "ndof: " << track->ndof << endl;
	exit(3);
      }
      result.incrementValue(totWeight, weight);
      planeWeight += weight;
      if(real){
	result.incrementValue(realWeight,weight);
	result.incrementCount(nMeas);
	if( not included ){
	  result.incrementCount(nMissed);
	}
      } else {
	result.incrementValue(fakeWeight,weight);
	planeFakeWeight += weight;
      }
    }
  }
  if(planeFakeWeight > 0.5 * planeWeight ){
    result.incrementCount(nAdditionalGhosts);
  }
}

//Function that performs 1/nThread of the simulation + analysis
void job(TrackerSystem<float,4>* bigSys, Results* result,  int nTracks, int nNoise, float efficiency){

  //Copy trackersystem to avoid problems with concurrency
  TrackerSystem<float,4> system((*bigSys));
  
  //The simulated measurements are stored here
  std::vector< std::vector<Measurement<float> > > simTracks;
  //The true states of the simulated particle is stored here
  
  //Write to result 2 to avoid mutex locking the analysis
  Results tmp;
  
  for(int event = 0; event < nTracks ;event++){
    simTracks.clear();
    system.clear();
    
    simulate(system, simTracks, efficiency, tmp.truth);
    //Add nNoise noise hits per plane
    for(size_t pl = 0; pl < system.planes.size(); pl++){
      boost::mutex::scoped_lock lock(randGuard);
      for(int noise = 0; noise < nNoise; noise ++){
	float x = (normRand() - 0.5) * 5000.0;
	float y = (normRand() - 0.5) * 5000.0;
	system.addMeasurement(pl, x, y, system.planes.at(pl).getZpos(), false, pl);
      }
      //Shuffle the measurement vector to avoid any systematic advantages for the track finder.
      random_shuffle( system.planes.at(pl).meas.begin(), system.planes.at(pl).meas.begin());
    }
    analyze(system, simTracks, tmp);
  }
  result->increment(tmp);
}

int main(){

  double ebeam = 100.0;
  int nPlanes = 9;
  
  TrackerSystem<float,4> system;
  //Configure track finder, these cuts are in place to let everything pass
  system.setChi2OverNdofCut( 6.0f); //Chi2 / ndof cut for the combinatorial KF to accept the track
  system.setNominalXdz(0.0f); //Nominal beam angle
  system.setNominalYdz(0.0f); //Nominal beam angle
  system.setXdzMaxDeviance(0.0005f);//How far og the nominal angle can the first two measurements be?
  system.setYdzMaxDeviance(0.0005f);//How far og the nominal angle can the first two measurements be?
 
  //Add planes to the tracker system
  float scattertheta = getScatterSigma(ebeam,.1);  
  float scattervar = scattertheta * scattertheta;
  system.addPlane(0, 10    , 4.3f, 4.3f,  scattervar, false);
  system.addPlane(1, 150010, 4.3f, 4.3f,  scattervar, false);
  system.addPlane(2, 300010, 4.3f, 4.3f,  scattervar, false);
  system.addPlane(3, 361712, 4.3f, 4.3f,  scattervar, true);
  system.addPlane(4, 454810, 4.3f, 4.3f,  scattervar, true);
  system.addPlane(5, 523312, 4.3f, 4.3f,  scattervar, true);
  system.addPlane(6, 680010, 4.3f, 4.3f,  scattervar, false);
  system.addPlane(7, 830010, 4.3f, 4.3f,  scattervar, false);
  system.addPlane(8, 980010, 4.3f, 4.3f,  scattervar, false);
  system.init();
  
  int nTracks = 1000000; //Number of tracks to be simulated
  int nThreads = 4;
  float efficiency = 0.95;
 
  srandom ( time(NULL) );
  std::srand ( unsigned ( std::time(0) ) );

  char hashname[200];

  //Default cut parameters
#ifdef DAF
  float chi2cut = 5.0;
#else
  float chi2cut = 0.0;
#endif
  float ckfcut = 6;
  int radius = 0;

  //Scan cut parameter for selected method
#ifdef CLU
  for(radius = 60; radius < 200; radius += 5)
#elif defined DAF
  for(chi2cut = 1; chi2cut < 20 ; chi2cut += 0.5)
#else
  for(ckfcut = 1; ckfcut < 20 ; ckfcut += 0.5)
#endif
  {
    //Prepare lisp output to be analyzed elsewhere.
    //Create hash tabke name from cut values
#ifdef CLU
    sprintf(hashname,"*noisehash-rad-%d*", radius);
#else
    sprintf(hashname,"*noisehash-%d-%d*", int(rint(2.0 * chi2cut)), int(rint(2.0 * ckfcut)));
#endif
    cout << "(defparameter " << hashname <<" (make-hash-table :test #\'equalp))" << endl;
    
    //Prepare track finder
    system.setCKFChi2Cut( ckfcut * ckfcut ); //Cut on the chi2 increment for the inclusion of a new measurement for combinatorial KF 
    system.setDAFChi2Cut( chi2cut * chi2cut ); // DAF chi2 cut-off
    system.setClusterRadius(radius);
    
    //Print the real values for the cuts to lisp
    cout << ";;;daf-chi2: " << system.getDAFChi2Cut() << endl;
    cout << ";;;ckf-chi2: " << system.getCKFChi2Cut() << endl;
    cout << ";;;radius: "   << radius << endl;

    //Reset counters
    for(int noise = 0; noise < 21; noise ++){
      Results result;
      //Start simulation + analysis job
      std::vector< std::vector<Measurement<float> > > simTracks;
      boost::thread_group threads;
      for(int ii = 0; ii < nThreads; ii++){
	threads.create_thread( boost::bind(job, &system, &result, nTracks / nThreads, noise, efficiency));
      }
      threads.join_all();
      
      //Write results as lisp
      result.printPlist(noise, hashname);
    }
  }
  //Print the algorithm type
#ifdef CLU
  cout << ";;;CLU!!!" << endl;
#elif defined  DAF
  cout << ";;;DAF!!!" << endl;
#else
  cout << ";;;KF!!!" << endl;
#endif
  
  return(0);
}
