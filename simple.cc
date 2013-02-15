#include <cmath>
#include "EUTelDafTrackerSystem.h"
#include <TH1D.h>
#include <TFile.h>
/*
Simnple application og the straight line track fitter
*/

double normRand(){
  return( (double) random() / (double) RAND_MAX);
}

void gaussRand(double& x1, double& x2){
  double w;
  while(true){
    x1 = 2.0f * normRand() -1.0f;
    x2 = 2.0f * normRand() -1.0f;
    w = x1 * x1 + x2 * x2;
    if( w < 1.0f) { break; }
  }
  w = std::sqrt( (-2.0f * std::log(w))/w);
  x1 *= w;
  x2 *= w;
}

double getScatterSigma(double eBeam, double radLength){
  radLength = fabs(radLength);
  double scatterTheta = 0.0136f/ eBeam * sqrt( radLength ) *  (1.0f + 0.038f * std::log(radLength) );
  return(scatterTheta);
}

int main(){
  double ebeam = 100.0;
  int nPlanes = 9;

  //Initialize histograms
  vector <TH1D*> resX;
  vector <TH1D*> resY;
  TH1D* chi2  = new TH1D("chi2","chi2",100,0,50);
  for(int ii = 0; ii < nPlanes; ii++){
    char name[100];
    sprintf(name, "resX%i", ii);
    resX.push_back( new TH1D(name,name,100, -50, 50));
    sprintf(name, "resY%i", ii);
    resY.push_back( new TH1D(name,name,100, -50, 50));
  }

  TrackerSystem<float,4> system;
  //Configure track finder, these cuts are in place to let everything pass
  system.setCKFChi2Cut(1000.0f); //Cut on the chi2 increment for the inclusion of a new measurement for combinatorial KF 
  system.setChi2OverNdofCut( 1000.0f); //Chi2 / ndof cut for the combinatorial KF to accept the track
  system.setNominalXdz(0.0f); //Nominal beam angle
  system.setNominalYdz(0.0f); //Nominal beam angle
  system.setXdzMaxDeviance(1.0f);//How far og the nominal angle can the first two measurements be?q
  system.setYdzMaxDeviance(1.0f);//How far og the nominal angle can the first two measurements be?
  system.setDAFChi2Cut( 100.0f); // DAF chi2 cut-off
  
  //Add planes to the tracker system
  float scattertheta = getScatterSigma(ebeam,.1);  
  float scattervar = scattertheta * scattertheta;
  system.addPlane(0, 10    , 4.3f, 4.3f, scattervar,  false);
  system.addPlane(1, 150010, 4.3f, 4.3f,  scattervar, false);
  system.addPlane(2, 300010, 4.3f, 4.3f,  scattervar, false);
  system.addPlane(3, 361712, 4.3f, 4.3f,  scattervar, false);
  system.addPlane(4, 454810, 4.3f, 4.3f,  scattervar, false);
  system.addPlane(5, 523312, 4.3f, 4.3f,  scattervar, false);
  system.addPlane(6, 680010, 4.3f, 4.3f,  scattervar, false);
  system.addPlane(7, 830010, 4.3f, 4.3f,  scattervar, false);
  system.addPlane(8, 980010, 4.3f, 4.3f,  scattervar, false);
  system.init();
  
  int nTracks = 1000000; //Number of tracks to be simulated

  for(int ii = 0; ii < nTracks ;ii++){
    system.clear();

    //simulate tracks with scattering
    double x(0.0f), y(0.0f), dx(0.0f), dy(0.0f);
    gaussRand(x, y);
    x *= 10000.0;
    y *= 10000.0;
    double g1, g2;
    gaussRand(g1, g2);
    dx += g1 * 0.0001;
    dy += g2 * 0.0001;
    double zPos = 0;
    for(size_t pl = 0; pl < nPlanes; pl++){
      double zDistance = system.planes.at(pl).getZpos() - zPos;
      zPos = system.planes.at(pl).getZpos();
      x += dx * zDistance;
      y += dy * zDistance;
      gaussRand(g1, g2);
      dx += g1 * getScatterSigma(ebeam, 0.1);
      dy += g2 * getScatterSigma(ebeam, 0.1);
      gaussRand(g1, g2);
      //Add a measurement to the tracker system at plane pl
      system.addMeasurement(pl, x + g1 * 4.3f, y + g2 * 4.3f, system.planes.at(pl).getZpos(), true, pl);
    }

    //Track finder
    system.combinatorialKF();
    //Loop over track candidates
    for(size_t ii = 0; ii < system.getNtracks(); ii++ ){
      if( ii > 0) { cout << "Several candidates!!!" << endl; }
      TrackCandidate<float, 4>* track =  system.tracks.at(ii);
      //Prepare track candidate for DAF fitting by assigning weights to measurements.
      system.indexToWeight( track );
      //Fit track with DAF. Also calculates DAF approximation of chi2 and ndof
      system.fitPlanesInfoDaf( track );
      //Extract the indexes from DAF weights for easy plotting
      system.weightToIndex( track );
      //Fill plots
      chi2->Fill(track->chi2);
      for(int pl = 0; pl < system.planes.size(); pl++){
	//Index of measurement used by track
	int index = track->indexes.at(pl);
	//If index is -1, no measurement was used in the plane
	if( index < 0) {continue;}
	//               x position of track at plane pl  - x position of measurement with index index at plane pl
	resX.at(pl)->Fill( track->estimates.at(pl)->getX() - system.planes.at(pl).meas.at(index).getX()  );
	resY.at(pl)->Fill( track->estimates.at(pl)->getY() - system.planes.at(pl).meas.at(index).getY()  );
      }
    }
  }

  //Save plots to a root file
  TFile* tfile = new TFile("plots/simple.root", "RECREATE");
  chi2->Write();
  for( int ii = 0; ii < system.planes.size(); ii++) {
    resX.at( ii )->Write();
    resY.at( ii )->Write();
  }
  cout << "Plotting to plots/simple.root" << endl;
  tfile->Close();
  
  return(0);
}

