#include <cmath>
#include <TH1D.h>
#include <TFile.h>
#include <map>
#include <TMath.h>

#include <boost/thread.hpp>
#include <boost/thread/detail/thread_group.hpp>
#include <boost/bind.hpp>

boost::mutex plotGuard;
boost::mutex randGuard;

#include "EUTelDafTrackerSystem.h"

#include "simutils.h"

//#define DAF
//#define HISTOS

/*
Simnple application og the straight line track fitter
*/
enum histoBase{
  chi2 = 0,
  chi2overndof = 1,
  //ndof = 2,
  resX = 10,
  resY = 20,
  pullX = 30,
  pullY = 40,
  tpullx = 50,
  tpulldx = 60,
};


// counter: nMeas, nMissed, nSuccess, nGhost, nAccepted, nEvents;
// value: totWeight, realWeight, fakeWeight

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

  ndof = 11,
  ndofsq = 12,
  ghostndof = 13,
  ghostndofsq = 14,
};

// enum valueBase{
//   totTracks = 0, //Number of tracks found
//   totWeight = 1, //Total weight of all measurements in track
//   noiseWeight = 2, //Weight assigned to noise hits
//   nMissed = 3, //Number of hits not included in fit
//   nMeasSim = 4, //Number of simulated hits
//   nMeasFound = 5, //Number of found hits
//   nMeasTot = 6, //Number of found hits
//   acceptedTracks = 7, //Number of tracks found
//   nGhosts = 8, //Number of ghost tracks
// };

class Histo2d {
  vector<vector< double> > histo;
  vector<double> chi2Real;
  vector<double> chi2Fake;
  vector<double> chi2Track;
  vector<double> pull_x, pull_dx;
  vector<double> trackPval;
public:
  Histo2d(){
    chi2Real.resize(110,0.0);
    chi2Fake.resize(110,0.0);
    chi2Track.resize(110,0);
    histo.resize(110);
    pull_x.resize(100,0);
    pull_dx.resize(100,0);
    trackPval.resize(25,0);
    for(int ii = 0; ii < 110; ii++){
      histo.at(ii).resize(110,0);
    }
  }
  void fillTrack(TrackEstimate<float,4>& truth, TrackEstimate<float,4> *estimate){
    //Store weight matrix in truth estimate
    Matrix<float,4,4> weight = estimate->cov;
    fastInvert(weight);
    Matrix<float,4,1> diff = truth.params - estimate->params;
    float chi2 = (diff.transpose() * weight * diff)(0,0);
    if(isnan(chi2)){
      cout << "chi2: " << chi2 << endl;
    }
    if( chi2 < 27.5){
      chi2Track.at(floor(chi2 * 4))++;
    }
    double pVal = 1.0 - TMath::Gamma( 2, chi2 / 2.0);
    if(pVal < 1){
      trackPval.at( floor( pVal * 25) )++;
    } else {
      trackPval.at(24)++;
    }
    int index =  (5 * (diff(0)/sqrt( estimate->cov(0,0) ))) + 50;
    if(index >= 0 and index < 100){
      pull_x.at(index)++;
    }
    index = (5 * (diff(2)/sqrt( estimate->cov(2,2) ))) + 50;
    if(index >= 0 and index < 100){
      pull_dx.at(index)++;
    }
  }
  void fill(double chi2, double weight, bool real, TrackCandidate<float,4>* cnd, int plane){
    if(chi2 < 27.5){
      if(real){
	chi2Real.at(floor(chi2 * 4)) += 1.0;
      } else {
	chi2Fake.at(floor(chi2 * 4)) += 1.0;
      } 
      histo.at( floor(chi2 * 4)).at(floor(weight * 100)) += 1.0;
    }
    // if((chi2 < 5) and (weight < 0.05)){
    //   cout << "chi2: " << chi2 << endl;
    //   cout << "Weights: " << endl  << cnd->weights.at(plane) << endl;
    // }
  }
  void print(int plane){
    cout << "(defparameter *histo-" << plane << "*" << endl;
    cout << "(let ((content (make-array (list 110 110) :element-type \'double-float :initial-element 0.0d0)))" << endl;
    for (int ii = 0; ii < 110; ii++){
      for (int jj = 0; jj < 110; jj++){
	cout << "(setf (aref content " << ii << " " << jj <<") (coerce "
	     << histo.at(ii).at(jj) << " \'double-float))" << endl;
      }
    }
    cout << "(list :x-min 0 :x-bin-size 0.25 :x-nbin 110 :y-min 0 :y-bin-size 0.01 :y-nbin 110 :data content)))" << endl << endl;
    cout << "(defparameter *realchi-" << plane << "* (list :min 0 :bin-size 0.25 :data (list" << endl;
    for (int ii = 0; ii < 110; ii++){
      cout << " " << chi2Real.at(ii);
    }
    cout << ")))" << endl << endl;;
    
    cout << "(defparameter *fakechi-" << plane << "* (list :min 0 :bin-size 0.25 :data (list" << endl;
    for (int ii = 0; ii < 110; ii++){
      cout << " " << chi2Fake.at(ii);
    }
    cout << ")))" << endl << endl;

    cout << "(defparameter *trackchi-" << plane << "* (list :min 0 :bin-size 0.25 :data (list" << endl;
    for (int ii = 0; ii < 110; ii++){
      cout << " " << chi2Track.at(ii);
    }
    cout << ")))" << endl << endl;

    cout << "(defparameter *track-pullx-" << plane << "* (list :min -10 :bin-size 0.2 :data (list" << endl;
    for (int ii = 0; ii < 100; ii++){
      cout << " " << pull_x.at(ii);
    }
    cout << ")))" << endl << endl;

    cout << "(defparameter *track-pulldx-" << plane << "* (list :min -10 :bin-size 0.2 :data (list" << endl;
    for (int ii = 0; ii < 100; ii++){
      cout << " " << pull_dx.at(ii);
    }
    cout << ")))" << endl << endl;

    cout << "(defparameter *track-pval-" << plane << "* (list :min 0 :bin-size 0.04 :data (list" << endl;
    for (int ii = 0; ii < 25; ii++){
      cout << " " << trackPval.at(ii);
    }
    cout << ")))" << endl << endl;
  }
};

vector<Histo2d> histos;

#include "simfunction.h" 

void analyze(TrackerSystem<float,4>& system, std::vector< std::vector<Measurement<float> > >& simTracks, 
	      map<int, TH1D*>& histoMap,map<int,double> &valueMap,map<int,int> &counterMap,
	      vector< TrackEstimate<float,4> >& truth){
  system.combinatorialKF();
  //system.clusterTracker();
  int myTrack = -1;
  double chi2ndof = 1000;
  for(size_t ii = 0; ii < system.getNtracks(); ii++ ){
    TrackCandidate<float, 4>* track =  system.tracks.at(ii);
    //Prepare track candidate for DAF fitting by assigning weights to measurements.
    //Fit track with DAF. Also calculates DAF approximation of chi2 and ndof
#ifdef DAF
    system.indexToWeight( track );
    system.fitPlanesInfoDaf( track );
    system.weightToIndex( track );
#else
    system.fitPlanesInfoUnBiased( track );
    system.indexToWeight( track );
#endif
    //Extract the index for the minimum chi2/ndof track
    if( ((track->chi2 / track->ndof) < chi2ndof) ){
      //(track->ndof > 1.5)){ //and
      chi2ndof = track->chi2 / track->ndof;
      myTrack = ii;
    }
  }
  
  {
    boost::mutex::scoped_lock lock(plotGuard);
    
    counterMap[nEvents]++;
    if(myTrack == -1){ return;  }
    TrackCandidate<float, 4>* track =  system.tracks.at(myTrack);
    //Make a cut in the number of measurements included
    if(track->ndof < 3.5){ return; }
    if( (track->chi2 / track->ndof) > system.getChi2OverNdofCut()) { return; }
    if( isnan(track->chi2 / track->ndof) ){
      cout << "NAN CHI!" << endl;
      return;
    }

    counterMap[nAccepted]++;

    bool ghost = true;
    bool nanp = false;

    for(int ii = 0; ii < track->indexes.size(); ii++ ){
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
      valueMap[ghostndof] += track->ndof;
      valueMap[ghostndofsq] += track->ndof * track->ndof;
      //histoMap[ghostndof]->Fill(track->ndof);
      return;
    }
    //Now we have a good track, that is not a ghost track
    counterMap[nSuccess]++;
    histoMap[ndof]->Fill(track->ndof);
    valueMap[ndof] += track->ndof;
    valueMap[ndofsq] += track->ndof * track->ndof;
    
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

// void analyze(TrackerSystem<float,4>& system, std::vector< std::vector<Measurement<float> > >& simTracks, 
// 	     map<int, TH1D*>& histoMap,map<int,double> &valueMap,map<int,int> &counterMap,
// 	     vector< TrackEstimate<float,4> >& truth){
//   //Track finder
//   system.combinatorialKF();
//   //system.truthTracker();
//   //system.clusterTracker();
//   //Loop over track candidates
  
//   int myTrack = -1;
//   double chi2ndof = 1000;

//   for(size_t ii = 0; ii < system.getNtracks(); ii++ ){
//     TrackCandidate<float, 4>* track =  system.tracks.at(ii);
//     //Prepare track candidate for DAF fitting by assigning weights to measurements.
//     //Fit track with DAF. Also calculates DAF approximation of chi2 and ndof

// #ifdef DAF
//     system.indexToWeight( track ); //REMEMBER ME
//     system.fitPlanesInfoDaf( track );
//     system.weightToIndex( track );
// #else
//     system.fitPlanesInfoUnBiased( track );
//     system.indexToWeight( track );
// #endif
//     //Extract the index for the minimum chi2/ndof track
//     if( ((track->chi2 / track->ndof) < chi2ndof) ){
//         //(track->ndof > 1.5)){ //and
//       chi2ndof = track->chi2 / track->ndof;
//       myTrack = ii;
//     }
//   }
//   {
//     boost::mutex::scoped_lock lock(plotGuard);
//     for(int pl = 0; pl < system.planes.size(); pl++){
//       //Don't carte about excluded planes here.
//       if(system.planes.at(pl).isExcluded()) {continue;}
//       for(int meas = 0; meas < system.planes.at(pl).meas.size(); meas++){
// 	//If the measurement is marked as being in the good region, it is "real"
// 	bool real = system.planes.at(pl).meas.at(meas).goodRegion();
// 	//Count every measurement that is real
// 	if(real){ counterMap[nMeasSim]++;}
// 	//If the measurement is real, but no track is found, mark it as missed here
// 	if(real and myTrack < 0){ counterMap[nMissed] ++;}
//       }
//     }
//     if(myTrack >= 0){
//       TrackCandidate<float, 4>* track =  system.tracks.at(myTrack);
//       //Count all the tracks.
//       counterMap[totTracks] ++;
//       //Make a cut in the number of measurements included
//       if(track->ndof < 1.5){ return; }
//       if( (track->chi2 / track->ndof) > system.getChi2OverNdofCut()) { return; }
//       //Count tracks that pass
//       counterMap[acceptedTracks] ++;
//       //Plot chi2
//       histoMap[chi2]->Fill(track->chi2);
//       histoMap[chi2overndof]->Fill(track->chi2 / track->ndof);

//       Matrix<float, 2, 1> residuals;
//       Matrix<float, 2, 1> reserror;

//       bool ghost = true;
      
//       //Loop over all planes
//       for(int pl = 0; pl < system.planes.size(); pl++){
// #ifdef HISTOS
// 	histos.at(pl).fillTrack( truth.at(pl), track->estimates.at(pl) );
// #endif	
// 	//Don't care about DUT planes.
// 	if(system.planes.at(pl).isExcluded()) {continue;}
// 	//Index of measurement used by track
// 	int index = track->indexes.at(pl);
// 	//Loop over measurements
// 	for(int meas = 0; meas < system.planes.at(pl).meas.size(); meas++){
// 	  //float weight = system.planes.at(pl).weights(meas);
// 	  float weight = track->weights.at(pl)(meas);
// 	  //Is this measurement included in the track. One or less measurements can be included.
// 	  bool included = (meas == index);
// 	  //Use the DAF chi2 cut to see if we really are using KF
// 	  if(system.getDAFChi2Cut() < 0.001){
// 	    if(included){ //If KF, use indexes to assign weight to 0 or 1
// 	      weight = 1.0;
// 	    } else {
// 	      weight = 0.0;
// 	    }
// 	  }
// 	  //Is weight in the legal range? If not, skip event to avoid. Print warnings.
// 	  if(not ( weight <= 1.0 and weight >= 0.0)){
// 	    if(valueMap[totTracks] > 0){
// 	      cout << weight << " weight" << endl;
// 	      cout << "n track: " << valueMap[totTracks] << endl;
// 	      cout << "tot weight: " << valueMap[totWeight] << endl;
// 	      cout << track->chi2 << " , " << track->ndof << endl;
// 	    }
// 	    continue;
// 	  }

// 	  residuals = system.getResiduals(system.planes.at(pl).meas.at(meas), track->estimates.at(pl));
// 	  reserror = system.getUnBiasedResidualErrors(system.planes.at(pl), track->estimates.at(pl));
	  
// 	  double chi2M = residuals(0) * residuals(0)/reserror(0) + residuals(1) * residuals(1)/reserror(1);
	  
// 	  bool real = system.planes.at(pl).meas.at(meas).goodRegion();

// #ifdef HISTOS
// 	  histos.at(pl).fill(chi2M, track->weights.at(pl)(meas), real, track, pl);
// #endif
// 	  //Count the total weight
// 	  valueMap[totWeight] += weight;
// 	  //Count the number of measurements that are in used tracks(not interesting)
// 	  counterMap[nMeasTot] += 1;
// 	  //Is the measurement real? Chack with the region that is assigned in simulation
// 	  if(not real){ //Count weight of noise hits. Good tracks, not real hits. 
// 	    valueMap[noiseWeight] += weight;
// 	  } 
// 	  if(real and included){ //Count the number of real, included measurements (not interesting)
// 	    counterMap[nMeasFound] ++;
// 	    ghost = false;
// 	  }
// 	  if(real and not(included)){ //Missed hits. Real hits that are not included in the track.
// 	    counterMap[nMissed] ++;
// 	  }
// 	}
// 	//If index is -1, no measurement was used in the plane
// 	if( index < 0) {continue;}
// 	//Plot stuff
// 	//               x position of track at plane pl  - x position of measurement with index index at plane pl
// 	char* name = new char[100];
	
// 	residuals = system.getResiduals(system.planes.at(pl).meas.at(index), track->estimates.at(pl));
// 	reserror = system.getUnBiasedResidualErrors(system.planes.at(pl), track->estimates.at(pl));
	
// 	// histoMap[resX + pl]->Fill( residuals(0) );
// 	// histoMap[resY + pl]->Fill( residuals(1) );
// 	histoMap[resX + pl]->Fill( track->estimates.at(pl)->getXdz() );
// 	histoMap[resY + pl]->Fill( track->estimates.at(pl)->getYdz() );
// 	histoMap[pullX + pl]->Fill( residuals(0)/sqrt(reserror(0)) );
// 	histoMap[pullY + pl]->Fill( residuals(1)/sqrt(reserror(1)) );
// 	delete name;
//       }
//       if(ghost){ counterMap[nGhosts]++;}
//     }
//   }
// }

void job(TrackerSystem<float,4>* bigSys, map<int, TH1D*>* histoMap, map<int, double>* valueMap,
	 map<int, int>* counterMap, int nTracks, int nNoise, float efficiency){
  //Copy trackersystem to avoid problems with concurrency
  TrackerSystem<float,4> system((*bigSys));

  std::vector< std::vector<Measurement<float> > > simTracks;
  vector< TrackEstimate<float,4> > truth;
  truth.resize(9);


  for(int event = 0; event < nTracks ;event++){
    simTracks.clear();
    system.clear();
    
    //Simulate a track
    simulate(system, simTracks, efficiency, truth);
    
    //Add nNoise noise hits per plane
    for(size_t pl = 0; pl < system.planes.size(); pl++){
      for(int noise = 0; noise < nNoise; noise ++){
	float x = (normRand() - 0.5) * 5000.0;
	float y = (normRand() - 0.5) * 5000.0;
	system.addMeasurement(pl, x, y, system.planes.at(pl).getZpos(), false, pl);
      }
    }
    analyze(system, simTracks, (*histoMap), (*valueMap), (*counterMap), truth);
  }
  // cout << "(list :bin-size 0.1 :min 0 :data (list";
  // for(int ii = 0; ii < system.m_chi2vals.size(); ii++){
  //   cout << " " << system.m_chi2vals.at(ii);
  // }
  // cout << "))" << endl;
}

double getSigmaPop(double sum, double sumsq, int n){
  double variance = (sumsq - (sum*sum)/n)/(n - 1);
  return(sqrt(variance));
}

int main(){
  double ebeam = 100.0;
  int nPlanes = 9;
  
  histos.resize(9);

  //g_ckfRealChi2 = new TH1D("real","real",100,0,100);
  //g_ckfFakeChi2 = new TH1D("fake","fake",100,0,100);

  //Initialize histograms
  map<int, TH1D*> histoMap;
  histoMap[chi2]  = new TH1D("chi2","chi2",100,0,50);
  histoMap[chi2overndof]  = new TH1D("chi2overndof","chi2overndof",100,0,20);
  for(int ii = 0; ii < nPlanes; ii++){
    char name[100];
    sprintf(name, "resX%i", ii);
    histoMap[resX + ii] = new TH1D(name,name,100, -0.001, 0.001);
    sprintf(name, "resY%i", ii);
    histoMap[resY + ii] = new TH1D(name,name,100, -0.001, 0.001);
    sprintf(name, "pullX%i", ii);
    histoMap[pullX + ii] = new TH1D(name,name,100, -5, 5);
    sprintf(name, "pullY%i", ii);
    histoMap[pullY + ii] = new TH1D(name,name,100, -5, 5);
  }
  histoMap[ndof] = new TH1D("ndof","ndof", 11, -0.5, 10.5);
  histoMap[ghostndof] = new TH1D("ghostndof","ghostndof", 11, -0.5, 10.5);

  map<int, double> valueMap;
  map<int, int> counterMap;

  TrackerSystem<float,4> system;
  //Configure track finder, these cuts are in place to let everything pass
  system.setChi2OverNdofCut( 6.0f); //Chi2 / ndof cut for the combinatorial KF to accept the track
  system.setNominalXdz(0.0f); //Nominal beam angle
  system.setNominalYdz(0.0f); //Nominal beam angle
  system.setXdzMaxDeviance(0.0005f);//How far og the nominal angle can the first two measurements be?
  system.setYdzMaxDeviance(0.0005f);//How far og the nominal angle can the first two measurements be?
  // system.setXdzMaxDeviance(0.0003f);//How far og the nominal angle can the first two measurements be?
  // system.setYdzMaxDeviance(0.0003f);//How far og the nominal angle can the first two measurements be?
  
  

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

#ifdef DAF
  float chi2cut = 10.0;
#else
  float chi2cut = 0.0;
#endif
  float ckfcut = 10;
  char hashname[200];
  int radius = 0;
#ifdef DAF
  for(float chi2cut = 1; chi2cut < 20 ; chi2cut += 0.5){
#else
  for(float ckfcut = 1; ckfcut < 20 ; ckfcut += 0.5){
  //{
#endif
    //for(int radius = 10; radius < 200; radius += 5){
    //{
    sprintf(hashname,"*noisehash-%d-%d*", int(rint(2.0 * chi2cut)), int(rint(2.0 * ckfcut)));
    //sprintf(hashname,"*noisehash-hardangle-%d-%d*", int(chi2cut), int(ckfcut));
    //sprintf(hashname,"*noisehash-rad-%d*", radius);
    cout << "(defparameter " << hashname <<" (make-hash-table :test #\'equalp))" << endl;
    system.setCKFChi2Cut( ckfcut * ckfcut ); //Cut on the chi2 increment for the inclusion of a new measurement for combinatorial KF 
    system.setDAFChi2Cut( chi2cut * chi2cut ); // DAF chi2 cut-off
    system.setClusterRadius(radius);
    cout << "(format t \"daf-chi2: ~a, ckf-chi2: ~a\" " << system.getDAFChi2Cut() << " " << system.getCKFChi2Cut() << ")" << endl;
    
    for(int noise = 0; noise < 21; noise ++){  
      // counter: nMeas, nMissed, nSuccess, nGhost, nAccepted, nEvents;
      // value: totWeight, realWeight, fakeWeight
      valueMap[totWeight] = 0.0f;
      valueMap[realWeight] = 0.0f;
      valueMap[fakeWeight] = 0.0f;
      valueMap[ndof] = 0.0f;
      valueMap[ndofsq] = 0.0f;
      valueMap[ghostndof] = 0.0f;
      valueMap[ghostndofsq] = 0.0f;

      counterMap[nMeas] = 0;
      counterMap[nMissed] = 0;
      counterMap[nEvents] = 0;
      counterMap[nAccepted] = 0;
      counterMap[nSuccess] = 0;
      counterMap[nGhost] = 0;
      counterMap[nNan] = 0;

      // valueMap[totWeight] = 0.0f;
      // valueMap[noiseWeight] = 0.0f;
      
      // counterMap[totTracks] = 0;
      // counterMap[acceptedTracks] = 0;
      // counterMap[nMissed] = 0;
      // counterMap[nMeasSim] = 0;
      // counterMap[nMeasFound] = 0;
      // counterMap[nMeasTot] = 0;
      // counterMap[nGhosts] = 0;
      
      std::vector< std::vector<Measurement<float> > > simTracks;
      boost::thread_group threads;
      for(int ii = 0; ii < nThreads; ii++){
	threads.create_thread( boost::bind(job, &system, &histoMap, &valueMap, &counterMap, nTracks / nThreads, noise, efficiency));
      }
      threads.join_all();

      // counter: nMeas, nMissed, nSuccess, nGhost, nAccepted, nEvents;
      // value: totWeight, realWeight, fakeWeight
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
	   << " :contamination " << 100.0 * valueMap[realWeight] / valueMap[totWeight] << endl
	   << " :missing " << 100.0 * counterMap[nMissed] / counterMap[nMeas] << endl
	   << " :efficiency " << 100.0 * counterMap[nSuccess] / counterMap[nEvents] << endl
	   << " :ndof " << valueMap[ndof]/counterMap[nSuccess] << endl
	   << " :ndofstd " <<  getSigmaPop(valueMap[ndof], valueMap[ndofsq], counterMap[nSuccess]) << endl
	   << " :ghostndof " << valueMap[ghostndof]/counterMap[nGhost] << endl
	   << " :ghostndofstd " <<  getSigmaPop(valueMap[ghostndof], valueMap[ghostndofsq], counterMap[nGhost]) << endl
	   << "))" <<endl;
      
      //cout << "(setf (gethash " << noise << " *noisehash-" << chi2cut<<"-" << ckfcut << "*)" << endl << "(list" << endl
      // cout << "(setf (gethash " << noise << " " << hashname << ")" << endl << "(list" << endl
      // 	   << "  :ntracks " << counterMap[totTracks] << endl 
      // 	   << "  :naccept " << counterMap[acceptedTracks] << endl 
      // 	   << "  :totweight " << valueMap[totWeight] << endl 
      // 	   << "  :noiseweight " << valueMap[noiseWeight] << endl 
      // 	   << "  :nMissed " << counterMap[nMissed] << endl 
      // 	   << "  :nMeasSim " << counterMap[nMeasSim] << endl 
      // 	   << "  :nMeasTot " << counterMap[nMeasTot] << endl 
      // 	   << "  :nGhosts " << counterMap[nGhosts] << endl 
      // 	   << "  :nMeasFound " << counterMap[nMeasFound] << "))"<< endl;
      
#ifdef HISTOS
      for(int ii = 0; ii < histos.size(); ii++){
	histos.at(ii).print(ii);
      }
#endif
    }
  }
  
  //job(system, histoMap, nTracks, tracksPerEvent, plotGuard);
  
  //Save plots to a root file
  char fname[] = "plots/multitrack.root";
  TFile* tfile = new TFile(fname, "RECREATE");
  //g_ckfFakeChi2->Write();
  //g_ckfRealChi2->Write();
  for(map<int,TH1D*>::iterator it = histoMap.begin(); it != histoMap.end(); it++){
    (*it).second->Write();
  }
  cout << "Plotting to "<< fname << endl;
  cout << (int)histoMap[chi2]->GetEntries() << " tracks accepted." << endl;
#ifdef DAF
  cout << "DAF!!!" << endl;
#else
  cout << "KF!!!" << endl;
#endif
  tfile->Close();
  
  return(0);
}

