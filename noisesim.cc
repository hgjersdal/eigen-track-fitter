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

enum valueBase{
  nMeas = 0,
  nMissed = 1,
  nEvents = 2,
  nAccepted = 3,
  nSuccess = 4,
  nGhost = 5,
  totWeight = 6,
  realWeight = 7,
  fakeWeight = 8,
  nNan = 9,
};

#include "simfunction.h" 

// Function that analyses an event in simulated data.
void analyze(TrackerSystem<float,4>& system, std::vector< std::vector<Measurement<float> > >& simTracks, 
	     map<int,double> &valueMap,map<int,int> &counterMap, vector< TrackEstimate<float,4> >& truth){
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
  
  {
    boost::mutex::scoped_lock lock(plotGuard);
    counterMap[nEvents]++;
    if(myTrack == -1){ return;  } // -1 means no track was found
    TrackCandidate<float, 4>* track =  system.tracks.at(myTrack);
    //Make a cut in the number of measurements included, and  chi2/ndof
    if(track->ndof < ndofmin){ return; } 
    if( (track->chi2 / track->ndof) > system.getChi2OverNdofCut()) { return; }
    if( isnan(track->chi2 / track->ndof) ){
      cout << "NAN CHI!" << endl; // Should never happen. If it does, a bug in the weigh calculation is a probable cause
      exit(2);
    }

    counterMap[nAccepted]++;

    bool ghost = true;
    bool nanp = false;

    for(int ii = 0; ii < track->indexes.size(); ii++ ){
    //Skip planes excluded by model or 
      if(system.planes.at(ii).isExcluded() ) { continue;}
      if(track->indexes.at(ii) < 0 ) {continue;}
      if( system.planes.at(ii).meas.at(track->indexes.at(ii) ).goodRegion()){
	ghost = false;
      }
      if( (track->weights.at(ii).size() > 0) and isnan(track->weights.at(ii).sum()) ){
	nanp = true;
      }
    }
    if( nanp ){
      counterMap[nNan]++;
      return;
    }
    if(ghost){
      counterMap[nGhost]++;
      return;
    }
    //Now we have a good track, that is not a ghost track
    counterMap[nSuccess]++;

    //weight v chi2
    Matrix<float, 2, 1> residuals;
    Matrix<float, 2, 1> reserror;

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
	  cout << weight << ", " << pl << ", " << mm << ", " << counterMap[nEvents] << endl;;
	  cout << "ndof: " << track->ndof << endl;
	  counterMap[nNan]++;
	  counterMap[nSuccess]--;
	  return;
	}
	valueMap[totWeight] += weight;
	if(real){
	  valueMap[realWeight] += weight;
	  counterMap[nMeas]++;
	  if( not included ){
	    counterMap[nMissed]++;
	  }
	} else {
	  valueMap[fakeWeight] += weight;
	}
      }
    }
  }
}


//Function that performs 1/nThread of the simulation + analysis
void job(TrackerSystem<float,4>* bigSys, map<int, double>* valueMap,
	 map<int, int>* counterMap, int nTracks, int nNoise, float efficiency){

  //Copy trackersystem to avoid problems with concurrency
  TrackerSystem<float,4> system((*bigSys));

  //The simulated measurements are stored here
  std::vector< std::vector<Measurement<float> > > simTracks;
  //The true states of the simulated particle is stored here
  vector< TrackEstimate<float,4> > truth;
  truth.resize(9);//One state per plane

  for(int event = 0; event < nTracks ;event++){
    simTracks.clear();
    system.clear();
    
    simulate(system, simTracks, efficiency, truth);
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
    analyze(system, simTracks, (*valueMap), (*counterMap), truth);
  }
}

int main(){

  double ebeam = 100.0;
  int nPlanes = 9;
  
  //Initialize counters
  map<int, double> valueMap;
  map<int, int> counterMap;

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
  float ckfcut = 5;
  int radius = 0;

  //Scan cut parameter for selected method
#ifdef CLU
  for(radius = 110; radius < 200; radius += 5)
#elif defined DAF
  for(chi2cut = 5; chi2cut < 20 ; chi2cut += 0.5)
#else
  for(ckfcut = 5; ckfcut < 20 ; ckfcut += 0.5)
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
      valueMap[totWeight] = 0.0f;
      valueMap[realWeight] = 0.0f;
      valueMap[fakeWeight] = 0.0f;
      
      counterMap[nMeas] = 0;
      counterMap[nMissed] = 0;
      counterMap[nEvents] = 0;
      counterMap[nAccepted] = 0;
      counterMap[nSuccess] = 0;
      counterMap[nGhost] = 0;
      counterMap[nNan] = 0;
      
      //Start simulation + analysis job
      std::vector< std::vector<Measurement<float> > > simTracks;
      boost::thread_group threads;
      for(int ii = 0; ii < nThreads; ii++){
	threads.create_thread( boost::bind(job, &system, &valueMap, &counterMap, nTracks / nThreads, noise, efficiency));
      }
      threads.join_all();
      
      //Avoid divide by zero. 0 is returned if no data is available.
      double purity = (valueMap[totWeight] == 0) ? 0 : 100.0 * valueMap[realWeight] / valueMap[totWeight];
      double missingp = (counterMap[nMeas] == 0)? 0 : 100.0 * counterMap[nMissed] / counterMap[nMeas];
      double effi = (counterMap[nEvents] == 0)? 0 : 100.0 * counterMap[nSuccess] / counterMap[nEvents];
      
      //Write results as lisp
      cout << "(setf (gethash " << noise << " " << hashname << ")" << endl << "(list" << endl
	   << " :nmeas " << counterMap[nMeas] << endl
	   << " :nmissed " << counterMap[nMissed] << endl
	   << " :nevents " << counterMap[nEvents] << endl
	   << " :naccepted " << counterMap[nAccepted] << endl
	   << " :nsuccess " << counterMap[nSuccess] << endl
	   << " :nghost " << counterMap[nGhost] << endl
	   << " :nnan " << counterMap[nNan] << endl
	   << " :totweight " << valueMap[totWeight] << endl
	   << " :realweight " << valueMap[realWeight] << endl
	   << " :fakeweight " << valueMap[fakeWeight] <<  endl
	   << " :purity " << purity << endl
	   << " :missing " << missingp << endl
	   << " :efficiency " << effi  << endl
	   << "))" <<endl;
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
