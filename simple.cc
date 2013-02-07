#include <cmath>
#include "EUTelDafTrackerSystem.h"
#include <TH1D.h>
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
  TrackerSystem<float,4> system;
  system.setCKFChi2Cut(20.0f);
  system.setNominalXdz(0.0f);
  system.setNominalYdz(0.0f);
  system.setChi2OverNdofCut( 100.0f);
  system.setDAFChi2Cut( 1000.0f);
  
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
  
  TH1D* histo = new TH1D("","",300,0,40);
  double ndof;

  int nTracks = 1000000;

  for(int ii = 0; ii < nTracks ;ii++){
    system.clear();

    //simulate
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
      system.addMeasurement(pl, x + g1 * 4.3f, y + g2 * 4.3f, system.planes.at(pl).getZpos(), true, pl);
    }
    //Track finder
    system.combinatorialKF();
    for(size_t ii = 0; ii < system.getNtracks(); ii++ ){
      //fit track
      system.indexToWeight(system.tracks.at(ii));
      system.fitPlanesInfoDaf(system.tracks.at(ii));
      //system.fitPlanesInfoBiased(system.tracks.at(ii));
      histo->Fill(system.tracks.at(ii)->chi2);
      ndof = system.tracks.at(ii)->ndof;
    }
  }
  cout << "(let ((histo (list :min 0 :bin-size " << histo->GetBinWidth(2) <<" :data (list ";
  for(int bin = 1; bin <= histo->GetNbinsX(); bin++){
    cout << " " << histo->GetBinContent(bin) / (histo->GetBinWidth(2) * nTracks);
  }
  cout << "))))" << endl;
  cout << "ndof " << ndof << endl;
  cout << "histo mean " << histo->GetMean() << "," << histo->GetNormFactor() << endl;
}

