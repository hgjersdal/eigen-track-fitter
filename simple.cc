#include <cmath>
#include "EUTelDafTrackerSystem.h"
#include <TH1D.h>
#include <TFile.h>
#include <TMath.h>
#include <TRandom3.h>


#include "simutils.h"

/*
Simnple application og the straight line track fitter.

Simulation with no noise hits, perfect detection efficiency in all planes.

Plot pulls, residuals, chi2
*/

void printHisto(TH1D* histo){
  cout << "(list" 
       <<  " :min " << histo->GetXaxis()->GetXmin()
       << " :bin-size " << histo->GetBinWidth( 0 ) << endl
       << ":data (list" << endl;
  for(int ii = 1; ii <= histo->GetNbinsX(); ii++){
    cout << " " << histo->GetBinContent( ii );
  }
  cout << endl << "))" << endl;
}


//Print histograms to lisp
void lispify(vector <TH1D*> &pulls, TH1D* chi2, TH1D* pvals, const char* name){
  std::cout << "(defparameter *" << name << "*" << endl;
  std::cout << "(list" << std::endl;
  std::cout << ":chi2 ";
  printHisto(chi2);

  std::cout << ":p-val ";
  printHisto(pvals);

  //Pull histograms
  for(int ii = 0; ii < pulls.size(); ii++){
    std::cout << ":histo" << ii << " ";
    printHisto(pulls.at(ii));
  }
  //Pull means
  std::cout << ":means (list";
  for(int ii = 0; ii < pulls.size(); ii++){
    cout << " " << pulls.at(ii)->GetMean();
  }
  cout << endl << ")" << endl;
  //Pull sigmas
  std::cout << ":sigmas (list";
  for(int ii = 0; ii < pulls.size(); ii++){
    cout << " " << pulls.at(ii)->GetRMS();
  }
  cout << endl << ")" << endl;

  std::cout << "))" << std::endl; //closes list and defparameter
}

int main(){
  TRandom3 rand(time(nullptr));

  double ebeam = 100.0;
  int nPlanes = 9;

  srandom(time(nullptr));

  //Initialize histograms
  vector <TH1D*> resX;
  vector <TH1D*> resY;
  vector <TH1D*> pullX;
  vector <TH1D*> pullY;
  TH1D* chi2  = new TH1D("chi2","chi2",100,0,50);
  TH1D* pvals  = new TH1D("pvals","pvals",100,0,1);
  TH1D* pvals2  = new TH1D("pvals2","pvals2",100,0,1);
  for(int ii = 0; ii < nPlanes; ii++){
    char name[100];
    sprintf(name, "resX%i", ii);
    resX.push_back( new TH1D(name,name,100, -50, 50));
    sprintf(name, "resY%i", ii);
    resY.push_back( new TH1D(name,name,100, -50, 50));
    sprintf(name, "pullX%i", ii);
    pullX.push_back( new TH1D(name,name,100, -5, 5));
    sprintf(name, "pullY%i", ii);
    pullY.push_back( new TH1D(name,name,100, -5, 5));
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
  float thickness = 0.01;
  float scattertheta = getScatterSigma(ebeam,thickness);  
  float scattervar = scattertheta * scattertheta;

  system.addPlane(0, 10    , 4.3f, 4.3f, scattervar,  false);
  system.addPlane(1, 150010, 4.3f, 4.3f,  scattervar, false);
  system.addPlane(2, 300010, 4.3f, 4.3f,  scattervar, false);
  system.addPlane(3, 361712, 50.0/sqrt(12), 400.0/sqrt(12),  scattervar, false);
  system.addPlane(4, 454810, 50.0/sqrt(12), 400.0/sqrt(12),  scattervar, false);
  system.addPlane(5, 523312, 50.0/sqrt(12), 400.0/sqrt(12),  scattervar, false);
  system.addPlane(6, 680010, 4.3f, 4.3f,  scattervar, false);
  system.addPlane(7, 830010, 4.3f, 4.3f,  scattervar, false);
  system.addPlane(8, 980010, 4.3f, 4.3f,  scattervar, false);

  system.init();
  
  int nTracks = 1000000; //Number of tracks to be simulated

  for(int event = 0; event < nTracks ;event++){
    system.clear();

    //simulate tracks with scattering
    double x(0.0f), y(0.0f), dx(0.0f), dy(0.0f);
    // gaussRand(x, y);
    // x *= 10000.0;
    // y *= 10000.0;
    x = normRand() * 10000.0f;
    y = normRand() * 10000.0f;
    double g1, g2;
    //gaussRand(g1 g2);
    g1 = rand.Gaus();
    g2 = rand.Gaus();
    dx += g1 * 0.0001;
    dy += g2 * 0.0001;
    double zPos = 0;
    for(size_t pl = 0; pl < nPlanes; pl++){
      double zDistance = system.planes.at(pl).getZpos() - zPos;
      zPos = system.planes.at(pl).getZpos();
      x += dx * zDistance;
      y += dy * zDistance;
      //gaussRand(g1, g2);
      g1 = rand.Gaus();
      g2 = rand.Gaus();

      dx += g1 * getScatterSigma(ebeam, thickness);
      dy += g2 * getScatterSigma(ebeam, thickness);
      //gaussRand(g1, g2);
      g1 = rand.Gaus();
      g2 = rand.Gaus();

      //Add a measurement to the tracker system at plane pl
      if( pl == 3 or pl == 4 or pl == 5){
	system.addMeasurement(pl, x + g1 * 50.0f/sqrt(12.0f), y + g2 * 400.0f/sqrt(12.0f), system.planes.at(pl).getZpos(), true, pl);
      } else {
	system.addMeasurement(pl, x + g1 * 4.3f, y + g2 * 4.3f, system.planes.at(pl).getZpos(), true, pl);
      }
    }

    //Track finder
    system.combinatorialKF();
    
    //Loop over track candidates
    for(size_t ii = 0; ii < system.getNtracks(); ii++ ){
      if( ii > 0) { cout << "Several candidates!!!" << endl; }
      TrackCandidate<float, 4>& track =  system.tracks.at(ii);
      //Fit track with DAF. Also calculates DAF approximation of chi2 and ndof
      system.fitPlanesInfoDaf( track );
      //Extract the indexes from DAF weights for easy plotting
      system.weightToIndex( track );
      //Fill plots
      chi2->Fill(track.chi2);
      pvals->Fill( TMath::Gamma( track.ndof / 2, track.chi2 / 2.0) );
      double chi2 = 0;
      for(int ii = 0; ii < track.ndof; ii++){
	double g1(0.0), g2(0.0);
	//gaussRand(g1,g2);
	g1 = rand.Gaus();
	g2 = rand.Gaus();

	chi2 += g2 * g2;
      }

      pvals2->Fill( TMath::Gamma( track.ndof / 2, chi2 / 2.0) );
      Eigen::Matrix<float, 2, 1> residuals, reserror;
      for(int pl = 0; pl < system.planes.size(); pl++){
	//Index of measurement used by track
	int index = track.indexes.at(pl);
	//If index is -1, no measurement was used in the plane
	if( index < 0) {continue;}
	//               x position of track at plane pl  - x position of measurement with index index at plane pl
	residuals = system.getResiduals(system.planes.at(pl).meas.at(index), track.estimates.at(pl));
	reserror = system.getUnBiasedResidualErrors(system.planes.at(pl), track.estimates.at(pl));

	resX.at(pl)->Fill( residuals[0] );
	resY.at(pl)->Fill( residuals[1] );

	pullX.at(pl)->Fill( residuals[0] / sqrt(reserror[0]) );
	pullY.at(pl)->Fill( residuals[1] / sqrt(reserror[1]) );
      }
    }
  }

  //Save plots to a root file
  TFile* tfile = new TFile("plots/simple.root", "RECREATE");
  chi2->Write();
  pvals->Write();
  pvals2->Write();
  for( int ii = 0; ii < system.planes.size(); ii++) {
    resX.at( ii )->Write();
    resY.at( ii )->Write();
    pullX.at( ii )->Write();
    pullY.at( ii )->Write();
  }

  cout << "Plotting to plots/simple.root" << endl;
  tfile->Close();

  lispify(pullX, chi2, pvals, "pullX");
  
  return(0);
}
