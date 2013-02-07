#include "estmat.h"
#include <gsl/gsl_multimin.h>
#include <Eigen/LU>
#include <Eigen/Cholesky>

double getScatterSigma(double eBeam, double radLength){
  //Highland formula
  radLength = fabs(radLength);
  double scatterTheta = 0.0136f/ eBeam * sqrt( radLength ) *  (1.0f + 0.038f * std::log(radLength) );
  return(scatterTheta);
}

void EstMat::addTrack( std::vector<Measurement<FITTERTYPE> > track){
  //Add a track to memory
  tracks.push_back(track);
}

void EstMat::readTrack(int track){
  //Read a track into the tracker system into memory
  for(size_t meas = 0; meas < tracks.at(track).size(); meas++){
    Measurement<FITTERTYPE>& m1 = tracks.at(track).at(meas);
    for(size_t ii = 0; ii < system.planes.size(); ii++){
      if( (int) m1.getIden() == (int) system.planes.at(ii).getSensorID()){
	double x = m1.getX() * ( 1.0 + xScale.at(ii)) + m1.getY() * zRot.at(ii);
	double y = m1.getY() * ( 1.0 + yScale.at(ii)) - m1.getX() * zRot.at(ii);
	x += xShift.at(ii);
	y += yShift.at(ii); 
	
	double z = m1.getZ();
	system.addMeasurement(ii, x, y, z, true, m1.getIden());
	break;
      }
    }
  }
}

FITTERTYPE Chi2::operator() (){
  //Get the global chi2 of the track sample
  double chi2(0.0);
  for(int track = 0; track < mat.itMax; track++){
    mat.system.clear();
    mat.readTrack(track);
    mat.system.clusterTracker();
    for(size_t ii = 0; ii < mat.system.getNtracks(); ii++ ){
      mat.system.weightToIndex(mat.system.tracks.at(ii));
      mat.system.fitInfoFWBiased(mat.system.tracks.at(ii));
      mat.system.getChi2BiasedInfo(mat.system.tracks.at(ii));
      chi2 += mat.system.tracks.at(ii)->chi2;
    }
  }
  return(chi2);
}

void EstMat::getExplicitEstimate(TrackEstimate<FITTERTYPE, 4>* estim){
  //Get the explicit estimates
  fastInvert(estim->cov);
  //Sparse multiply
  FITTERTYPE x(estim->params(0)), y(estim->params(1)), dx(estim->params(2)), dy(estim->params(3));
  Matrix<FITTERTYPE, 4, 1> offdiag, reverse(dx, dy, x, y);
  offdiag(0) = offdiag(2) = estim->cov(0,2);
  offdiag(1) = offdiag(3) = estim->cov(1,3);
  estim->params = estim->params.cwise() * estim->cov.diagonal() + offdiag.cwise() * reverse;

  // estim->params(0) = estim->cov(0,0) * x + estim->cov(0,2) * dx;
  // estim->params(1) = estim->cov(1,1) * y + estim->cov(1,3) * dy;
  // estim->params(2) = estim->cov(2,2) * dx+ estim->cov(2,0) * x;
  // estim->params(3) = estim->cov(3,3) * dy+ estim->cov(3,1) * y;
} 

FITTERTYPE SDR::operator() (){
  // Standardized residuals deviance from mean = 0 and variance = 1.
  // Either the difference between the FW and BW sates(SRD1=true),
  // the pull distributions in FW and BW (SDR2=true) or both (SRD1 and SDR2)
  TrackerSystem<FITTERTYPE,4>& system = mat.system;
  std::vector< double > sqrPullXFW(system.planes.size() - 2, 0.0 );
  std::vector< double > sqrPullXBW(system.planes.size() - 2, 0.0 );
  std::vector< double > sqrPullYFW(system.planes.size() - 2, 0.0 );
  std::vector< double > sqrPullYBW(system.planes.size() - 2, 0.0 );
  std::vector< std::vector<double> > sqrParams(system.planes.size() - 3, std::vector<double>(4 ,0.0) ); 
  
  int nTracks = 0;
  for(int track = 0; track < mat.itMax; track++){
    //prepare system for new track: clear system from prev go around, read track from memory, run track finder
    system.clear();
    mat.readTrack(track);
    system.clusterTracker();
    for(size_t ii = 0; ii < system.getNtracks(); ii++ ){
      //Transkate DAF track candidate to KF candidate
      system.weightToIndex(system.tracks.at(ii));
      //Run FW fitter, get p-values
      system.fitInfoFWBiased(system.tracks.at(ii));
      system.fitInfoBWUnBiased(system.tracks.at(ii));
      //Get explicit estimates
      TrackEstimate<FITTERTYPE, 4>* fw1 = system.m_fitter->forward.at(1);
      mat.getExplicitEstimate(fw1);

      for(size_t pl = 0; pl < system.planes.size() -2; pl++){
	TrackEstimate<FITTERTYPE, 4>* fw = system.m_fitter->forward.at(pl + 2);
	TrackEstimate<FITTERTYPE, 4>* bw = system.m_fitter->backward.at(pl);
	mat.getExplicitEstimate(fw);
	mat.getExplicitEstimate(bw);
      }
      //Chi2 increments FW
      for(size_t pl = 2; pl < system.planes.size(); pl++){
	//I'm using an information filter, need explicit state and covariance
	TrackEstimate<FITTERTYPE, 4>* result = system.m_fitter->forward.at(pl);
	Measurement<FITTERTYPE>& meas = system.planes.at(pl).meas.at(0);
	//Get residuals in x and y, squared
	Matrix<FITTERTYPE, 2, 1> resids = system.getResiduals(meas, result).cwise().square();
	//Get variance of residuals in x and y
	Matrix<FITTERTYPE, 2, 1> variance = system.getBiasedResidualErrors(system.planes.at(pl), result);
	Matrix<FITTERTYPE, 2, 1> pull2 = resids.cwise() / variance;
	sqrPullXFW.at(pl - 2) += pull2(0); 
	sqrPullYFW.at(pl - 2) += pull2(1); 
      }
      //Chi2 increments BW
      for(size_t pl = 0; pl < system.planes.size() - 2; pl++){
	TrackEstimate<FITTERTYPE, 4>* result = system.m_fitter->backward.at(pl);
	Measurement<FITTERTYPE>& meas = system.planes.at(pl).meas.at(0);
	Matrix<FITTERTYPE, 2, 1> resids = system.getResiduals(meas, result).cwise().square();
	Matrix<FITTERTYPE, 2, 1> variance = system.getUnBiasedResidualErrors(system.planes.at(pl), result);
	Matrix<FITTERTYPE, 2, 1> pull2 = resids.cwise() / variance;
	sqrPullXBW.at(pl) += pull2(0);
	sqrPullYBW.at(pl) += pull2(1);
      }
      //Difference in parameters
      if(not cholDec){
	for(size_t pl = 1; pl < system.planes.size() -2; pl++){
	  TrackEstimate<FITTERTYPE, 4>* fw = system.m_fitter->forward.at(pl);
	  TrackEstimate<FITTERTYPE, 4>* bw = system.m_fitter->backward.at(pl);
	  for(size_t param = 0; param < 4; param++){
	    double var = fw->cov(param,param) + bw->cov(param,param);
	    double res = fw->params(param) - bw->params(param);
	    sqrParams.at(pl -1).at(param) += res * res / var;
	  }
	}
      } else {
	//Attempt at getting pull values from cholesky decomposed covariance.
	for(size_t pl = 1; pl < system.planes.size() -2; pl++){
	  TrackEstimate<FITTERTYPE, 4>* fw = system.m_fitter->forward.at(pl);
	  TrackEstimate<FITTERTYPE, 4>* bw = system.m_fitter->backward.at(pl);
	  Matrix<double, 4, 1> tmpDiff =  (fw->params - bw->params).cast<double>();
	  Matrix<FITTERTYPE, 4, 4> tmpCov =  fw->cov + bw->cov;
	  
	  //Use doubles, else it fails
	  Eigen::LLT<Matrix4d> tmpChol;
	  tmpChol.compute(tmpCov.cast<double>());
	  tmpChol.matrixL().marked<Eigen::LowerTriangular>().solveTriangularInPlace(tmpDiff);
	  for(size_t param = 0; param < 4; param++){
	    sqrParams.at(pl -1).at(param) += tmpDiff(param) * tmpDiff(param);
	  }
	}
      }
      nTracks++;
    }
  }
  
  double varvar(0.0);
  if(SDR2){
    for( size_t pl = 0; pl < system.planes.size() - 2; pl++){
      double resvar = 1.0 - sqrPullXFW.at(pl)/(nTracks - 1);
      varvar += resvar * resvar;
      resvar = 1.0 - sqrPullYFW.at(pl)/(nTracks - 1);
      varvar += resvar * resvar;
      resvar = 1.0 - sqrPullXBW.at(pl)/(nTracks - 1);
      varvar += resvar * resvar;
      resvar = 1.0 - sqrPullYBW.at(pl)/(nTracks - 1);
      varvar += resvar * resvar;
    }
  }
  if(SDR1){
    for( int pl = 1; pl < system.planes.size() - 2; pl++){
      for(int param = 0; param < 4; param++){
	double resvar = 1.0 - (sqrParams.at(pl - 1).at(param) / (nTracks - 1));
	varvar += resvar * resvar;
      }
    }
  }
  // SDR3, is 1 + 2
  return(varvar);
}

FITTERTYPE FwBw::operator() (){
  // The min log likelihood of the state difference between the FW and BW fit.
  TrackerSystem<FITTERTYPE,4>& system = mat.system;
  
  double logL(0.0);
  std::vector< double > sqrPullXFW(system.planes.size() - 2, 0.0 );
  std::vector< double > sqrPullXBW(system.planes.size() - 2, 0.0 );
  std::vector< double > sqrPullYFW(system.planes.size() - 2, 0.0 );
  std::vector< double > sqrPullYBW(system.planes.size() - 2, 0.0 );
  std::vector< std::vector<double> > sqrParams(system.planes.size() - 3, std::vector<double>(4 ,0.0) ); 
  int nTracks = 0;

  //loop over tracks
  for(int track = 0; track < mat.itMax; track++){
    system.clear();
    mat.readTrack(track);
    system.clusterTracker();
    //Loop over track candidates
    for(size_t ii = 0; ii < system.getNtracks(); ii++ ){
      nTracks++;
      //Translate candidate from DAF to KF
      system.weightToIndex(system.tracks.at(ii));
      system.fitInfoFWBiased(system.tracks.at(ii));
      system.fitInfoBWUnBiased(system.tracks.at(ii));
      //Get explicit estimates
      TrackEstimate<FITTERTYPE, 4>* fw1 = system.m_fitter->forward.at(1);
      mat.getExplicitEstimate(fw1);
      for(size_t pl = 0; pl < system.planes.size() -2; pl++){
	TrackEstimate<FITTERTYPE, 4>* fw = system.m_fitter->forward.at(pl + 2);
	TrackEstimate<FITTERTYPE, 4>* bw = system.m_fitter->backward.at(pl);
	mat.getExplicitEstimate(fw);
	mat.getExplicitEstimate(bw);
      }
      Matrix<FITTERTYPE, 4, 4> cov;
      Matrix<FITTERTYPE, 4, 1> resids;
      //Difference in parameters
      for(size_t pl = 1; pl < system.planes.size() -2; pl++){
	TrackEstimate<FITTERTYPE, 4>* fw = system.m_fitter->forward.at(pl);
	TrackEstimate<FITTERTYPE, 4>* bw = system.m_fitter->backward.at(pl);
	resids = fw->params - bw->params;
	cov = fw->cov + bw->cov;
	double determinant = cov.determinant();

	fastInvert(cov);
	double exponent = (resids.transpose() * cov * resids)(0,0);
	logL -= log( determinant ) +  exponent;
      }
      //Chi2 increments FW
      for(size_t pl = 2; pl < system.planes.size(); pl++){
	//I'm using an information filter, need explicit state and covariance
	TrackEstimate<FITTERTYPE, 4>* result = system.m_fitter->forward.at(pl);
	Measurement<FITTERTYPE>& meas = system.planes.at(pl).meas.at(0);
	//Get residuals in x and y, squared
	Matrix<FITTERTYPE, 2, 1> resids = system.getResiduals(meas, result).cwise().square();
	//Get variance of residuals in x and y
	Matrix<FITTERTYPE, 2, 1> variance = system.getBiasedResidualErrors(system.planes.at(pl), result);
	Matrix<FITTERTYPE, 2, 1> pull2 = resids.cwise() / variance;
	sqrPullXFW.at(pl - 2) += pull2(0); 
	sqrPullYFW.at(pl - 2) += pull2(1); 
      }
      //Chi2 increments BW
      for(size_t pl = 0; pl < system.planes.size() - 2; pl++){
	TrackEstimate<FITTERTYPE, 4>* result = system.m_fitter->backward.at(pl);
	Measurement<FITTERTYPE>& meas = system.planes.at(pl).meas.at(0);
	Matrix<FITTERTYPE, 2, 1> resids = system.getResiduals(meas, result).cwise().square();
	Matrix<FITTERTYPE, 2, 1> variance = system.getUnBiasedResidualErrors(system.planes.at(pl), result);
	Matrix<FITTERTYPE, 2, 1> pull2 = resids.cwise() / variance;
	sqrPullXBW.at(pl) += pull2(0); 
	sqrPullYBW.at(pl) += pull2(1); 
      }
    }
  }
  if(isnan(logL)){
    cout << "FWBW: nan" << endl;
    mat.printParams( (char*) "params[\"RadiationLengths\"]", mat.radLengths, true);
    mat.printParams( (char*) "params[\"ResolutionX\"]", mat.resX, true);
    mat.printParams( (char*) "params[\"ResolutionY\"]", mat.resY, true);
  }
  retVal2 = 0.0;
  for( int pl = 0; pl < system.planes.size() - 2; pl++){
    double resvar = 1.0 - sqrPullXFW.at(pl)/(nTracks - 1);
    retVal2 += resvar * resvar;
    resvar = 1.0 - sqrPullYFW.at(pl)/(nTracks - 1);
    retVal2 += resvar * resvar;
    resvar = 1.0 - sqrPullXBW.at(pl)/(nTracks - 1);
    retVal2 += resvar * resvar;
    resvar = 1.0 - sqrPullYBW.at(pl)/(nTracks - 1);
    retVal2 += resvar * resvar;
  }
  //Minimize, not maximize
  return(-1.0 * logL);
}
  
double normRand(){
  //A random number between 0 and 1
  return( (double) random() / (double) RAND_MAX);
}

void gaussRand(double& x1, double& x2){
  //A random number following the normal distribution, mean 0, sigma 1
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

void EstMat::simulate(int nTracks){
  // Toy simulation of a straight track with Gaussian uncertainties and scattering
  cout << "Simulation parameters" << endl;
  printParams( (char*) "params[\"RadiationLengths\"]", radLengths, true);
  printParams( (char*) "params[\"ResolutionX\"]", resX, true);
  printParams( (char*) "params[\"ResolutionY\"]", resY, true);

  size_t nPlanes = system.planes.size();
  
  //tracks
  for( int track = 0; track < nTracks; track++){
    std::vector<Measurement<FITTERTYPE> > simTrack;
    if( track + 1 % 100000 == 0){
      std::cout << "Simulating track: " << track << std::endl;
    }
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
      dx += g1 * getScatterSigma(eBeam, radLengths.at(pl));
      dy += g2 * getScatterSigma(eBeam, radLengths.at(pl));
      gaussRand(g1, g2);
      Matrix<FITTERTYPE, 2, 1> sigmas;
      sigmas(0) = resX.at(pl);
      sigmas(1) = resY.at(pl);
      if( pl == 3 or pl == 4 or pl == 5){
	//double posX = x + (normRand() - 0.5) * 50;
	//double posY = y + (normRand() - 0.5) * 400;
	//simTrack.push_back( Measurement(posX, posY, system.planes.at(pl).getZpos(), true, pl) );
	simTrack.push_back( Measurement<FITTERTYPE>(x + g1 * sigmas(0), y + g2 * sigmas(1), system.planes.at(pl).getZpos(), true, pl) );
      } else {
	simTrack.push_back( Measurement<FITTERTYPE>(x + g1 * sigmas(0), y + g2 * sigmas(1), system.planes.at(pl).getZpos(), true, pl) );
      }
    }
    addTrack(simTrack);
  }
}

void EstMat::addPlane(FitPlane<FITTERTYPE>& pl){
  //Add a plane to the material estimator
  system.planes.push_back( FitPlane<FITTERTYPE>(pl) );
}

void EstMat::initSim(int nPlanes){
  //Initialize random number generators, tracker system, ...
  //seed with time
  srandom ( time(NULL) );

  system.setClusterRadius( 100000.0f);
  system.setNominalXdz(0.0f);
  system.setNominalYdz(0.0f);
  system.setChi2OverNdofCut( 100.0f);
  system.setDAFChi2Cut( 1000.0f);

  for(int ii = 0; ii < nPlanes; ii++){
    double scatterTheta = getScatterSigma(eBeam, 1.0);
    system.addPlane(ii, ii, 4.3f, 4.3f, scatterTheta * scatterTheta, false);
  }

  system.init();
  xShift.resize(nPlanes);
  yShift.resize(nPlanes);
}

void EstMat::movePlaneZ(int planeIndex, double zPos){
  //Move the plane in the z direction
  FitPlane<FITTERTYPE>& pl = system.planes.at(planeIndex);
  pl.setZpos( zPos);
  Matrix<FITTERTYPE, 3, 1> ref = pl.getRef0();
  ref(2) += zPos;
  pl.setRef0( ref );
}

void EstMat::setPlane(int index, double sigmaX, double sigmaY, double radLength){
  //Change the state of a plane
  double scatterTheta = getScatterSigma(eBeam, radLength);
  system.planes.at(index).setScatterThetaSqr( scatterTheta * scatterTheta);
  system.planes.at(index).setSigmas(sigmaX, sigmaY);
}

void EstMat::printParams(char* name, std::vector<FITTERTYPE>& params, bool plot){
  //Print estimation parameters to screen
  if(not plot){ return;}
  std::cout << name << " = \" ";
  for( size_t pos = 0; pos < params.size(); pos++){
    printf( " %4.6f", params.at(pos));
  }
  std::cout << " \"" << std::endl;
}

void EstMat::plot(char* fname){
  //Make, fill and save some plots. The full track sample is fitted for this
  for(size_t plane = 0; plane < system.planes.size(); plane ++){
    double scatterSigma = getScatterSigma(eBeam, radLengths.at(plane));
    system.planes.at(plane).setScatterThetaSqr( scatterSigma * scatterSigma);
  }
  
  TFile* tfile = new TFile(fname, "RECREATE");
  std::vector<TH1D*> resX, resY, pullX, pullY;
  std::vector<TH1D*> xFB, yFB, dxFB, dyFB, pValFW;
  TH1D* chi2 = new TH1D("chi2","chi2", 150, 0, 150);
  TH1D* ndof = new TH1D("ndof", "ndof", 15, 0, 15);
  TH1D* pValue = new TH1D("pValue", "pValue", 100,0,1);
  TH1D* chi2OverNeod = new TH1D("chi2ndof","chi2 over ndof", 100, 0, 20);
  for(int ii = 0; ii < (int) system.planes.size(); ii++){
    char name[200];
    int sensorID = system.planes.at(ii).getSensorID();
    sprintf(name, "resX %i", sensorID); 
    resX.push_back( new TH1D(name,name, 800, -400, 400));
    sprintf(name, "resY %i", sensorID); 
    resY.push_back( new TH1D(name,name, 800, -400, 400));
    sprintf(name, "pullX unbiased  %i", sensorID); 
    pullX.push_back( new TH1D(name,name, 100, -10, 10));
    sprintf(name, "pullY biased  %i", sensorID); 
    pullY.push_back( new TH1D(name,name, 100, -10, 10));
  }
  
  cout << "Inited plots" << endl;
  
  //Loop over all tracks
  for(size_t track = 0; track < tracks.size(); track++){
    system.clear();
    readTrack(track);
    system.clusterTracker();
    for(size_t ii = 0; ii < system.getNtracks(); ii++ ){
      TrackCandidate<FITTERTYPE, 4>* candidate = system.tracks.at(ii);
      system.weightToIndex(candidate);
      system.fitPlanesInfoUnBiased(candidate);
      //get p-val
      pValue->Fill( TMath::Gamma( candidate->ndof / 2, candidate->chi2 / 2.0) );
      chi2->Fill( candidate->chi2 );
      ndof->Fill( candidate->ndof );
      chi2OverNeod->Fill( candidate->chi2 / candidate->ndof );
      
      for(int ii = 0; ii < (int) system.planes.size() ; ii++ ){
	candidate->estimates.at(ii)->copy( system.m_fitter->smoothed.at(ii) );
      }
      for(size_t pl = 0; pl < system.planes.size(); pl++){
      	if( system.planes.at(pl).meas.size() < 1){ continue; }
      	Measurement<FITTERTYPE>& meas = system.planes.at(pl).meas.at(0);
	TrackEstimate<FITTERTYPE, 4>* est = candidate->estimates.at(pl);
	Matrix<FITTERTYPE, 2, 1> resids = system.getResiduals(meas, est);
	Matrix<FITTERTYPE, 2, 1> variance = system.getUnBiasedResidualErrors(system.planes.at(pl), est);
      	resX.at(pl)->Fill(resids(0));
      	resY.at(pl)->Fill(resids(1));
      	pullX.at(pl)->Fill(resids(0)/sqrt(variance(0)));
      	pullY.at(pl)->Fill(resids(1)/sqrt(variance(1)));
      }
    }
  }
  cout << "Saving " << endl;
  tfile->cd();
  chi2->Write();
  pValue->Write();
  ndof->Write();
  chi2OverNeod->Write();
  for(size_t ii = 0; ii < system.planes.size(); ii++){
    resX.at(ii)->Write();
    resY.at(ii)->Write();
    pullX.at(ii)->Write();
    pullY.at(ii)->Write();
  }
  cout << "Write" << endl;
  tfile->Write();
  cout << "Done plotting to " << fname << endl;
}

//simplex stuff
size_t EstMat::getNSimplexParams(){
  //The number of parameters to be estimated
  size_t params(0);
  params += resXIndex.size();
  params += resYIndex.size();
  params += radLengthsIndex.size();
  params += resXMulti.size();
  params += resYMulti.size();
  params += resXYMulti.size();
  params += xShiftIndex.size();
  params += yShiftIndex.size();
  params += xScaleIndex.size();
  params += yScaleIndex.size();
  params += zRotIndex.size();
  params += zPosIndex.size();
  return( params );
}

gsl_vector* EstMat::systemToEst(){
  //Read the state of the tracker system into a gsl vector for the simplex search
  int nParams = getNSimplexParams();
  gsl_vector* v = gsl_vector_alloc(nParams);

  double param = 0;
  for(size_t ii = 0; ii < resXIndex.size(); ii ++){
    gsl_vector_set(v, param++, resX.at( resXIndex.at(ii)) );
  }
  for(size_t ii = 0; ii < resYIndex.size(); ii ++){
    gsl_vector_set(v, param++, resY.at( resYIndex.at(ii)) );
  }
  for(size_t ii = 0; ii < radLengthsIndex.size(); ii ++){
    gsl_vector_set(v, param++, radLengths.at( radLengthsIndex.at(ii)) );
  }
  if( resXMulti.size() > 0){
    gsl_vector_set(v, param++, resX.at( resXMulti.at(0)) );
  }
  if( resYMulti.size() > 0){
    gsl_vector_set(v, param++, resY.at( resYMulti.at(0)) );
  }
  if( resXYMulti.size() > 0){
    gsl_vector_set(v, param++, resX.at( resXYMulti.at(0)) );
  }

  //Alignment
  for(size_t ii = 0; ii < xShiftIndex.size(); ii++){
    gsl_vector_set(v, param++, xShift.at( xShiftIndex.at(ii)));
  }
  for(size_t ii = 0; ii < yShiftIndex.size(); ii++){
    gsl_vector_set(v, param++, yShift.at( yShiftIndex.at(ii)));
  }
  for(size_t ii = 0; ii < xScaleIndex.size(); ii++){
    gsl_vector_set(v, param++, xScale.at( xScaleIndex.at(ii)));
  }
  for(size_t ii = 0; ii < yScaleIndex.size(); ii++){
    gsl_vector_set(v, param++, yScale.at( yScaleIndex.at(ii)));
  }
  for(size_t ii = 0; ii < zRotIndex.size(); ii++){
    gsl_vector_set(v, param++, zRot.at( zRotIndex.at(ii)));
  }
  for(size_t ii = 0; ii < zPosIndex.size(); ii++){
    gsl_vector_set(v, param++, zPos.at( zPosIndex.at(ii)));
  }
  return(v);
}

void EstMat::estToSystem( const gsl_vector* params){
  //Set the system state from the gsl estimation vector
  double param = 0;
  vector<int>::iterator it;
  for(it = resXIndex.begin(); it != resXIndex.end(); it++){
    resX.at((*it)) = gsl_vector_get(params, param++);
    system.planes.at((*it)).setSigmaX( resX.at((*it)));
  }
  for(it = resYIndex.begin(); it != resYIndex.end(); it++){
    resY.at((*it)) = gsl_vector_get(params, param++);
    system.planes.at((*it)).setSigmaY( resY.at((*it)));
  }
  for(it = radLengthsIndex.begin(); it != radLengthsIndex.end(); it++){
    radLengths.at((*it)) = gsl_vector_get(params, param++);
    double scatterTheta = getScatterSigma(eBeam, radLengths.at((*it)));
    system.planes.at((*it)).setScatterThetaSqr( scatterTheta * scatterTheta);
  }
  if( resXMulti.size() > 0){
   for(size_t ii = 0; ii < resXMulti.size(); ii++){
      int index = resXMulti.at(ii);
      resX.at(index) = gsl_vector_get(params, param++);
      system.planes.at(index).setSigmaX(resX.at(index));
    }
  }
  if( resYMulti.size() > 0){
    for(size_t ii = 0; ii < resYMulti.size(); ii++){
      int index = resYMulti.at(ii);
      resY.at(index) = gsl_vector_get(params, param++);
      system.planes.at(index).setSigmaY(resX.at(index));
    }
  }
  if( resXYMulti.size() > 0){
    for(size_t ii = 0; ii < resXYMulti.size(); ii++){
      int index = resXYMulti.at(ii);
      resX.at(index) = resY.at(index) = gsl_vector_get(params, param++);
      system.planes.at(index).setSigmas(resX.at(index), resY.at(index));
    }
  }

  //Alignment
  for(it = xShiftIndex.begin(); it != xShiftIndex.end(); it++){
    xShift.at((*it)) = (gsl_vector_get(params, param++));
  }
  for(it = yShiftIndex.begin(); it != yShiftIndex.end(); it++){
    yShift.at((*it)) = (gsl_vector_get(params, param++));
  }
  for(it = xScaleIndex.begin(); it != xScaleIndex.end(); it++){
    xScale.at((*it)) = (gsl_vector_get(params, param++));
  }
  for(it = yScaleIndex.begin(); it != yScaleIndex.end(); it++){
    yScale.at((*it)) = (gsl_vector_get(params, param++));
  }
  for(it = zRotIndex.begin(); it != zRotIndex.end(); it++){
    zRot.at((*it)) = (gsl_vector_get(params, param++));
  }
  for(it = zPosIndex.begin(); it != zPosIndex.end(); it++){
    zPos.at((*it)) = gsl_vector_get(params, param++);
    movePlaneZ((*it), zPos.at((*it)));
  }
}

gsl_vector* EstMat::simplesStepSize(){
  //Initial step sizes for the simplex search
  int nParams = getNSimplexParams();
  gsl_vector* s = gsl_vector_alloc(nParams);

  double param = 0;
  for(size_t ii = 0; ii < resXIndex.size(); ii ++){
    gsl_vector_set(s, param++, 0.1 * resX.at(resXIndex.at(ii)) );
  }
  for(size_t ii = 0; ii < resYIndex.size(); ii ++){
    gsl_vector_set(s, param++, 0.1 * resX.at(resYIndex.at(ii)) );
  }
  for(size_t ii = 0; ii < radLengthsIndex.size(); ii ++){
    gsl_vector_set(s, param++, 0.1 * radLengths.at(radLengthsIndex.at(ii)) );
  }
  if( resXMulti.size() > 0){
    gsl_vector_set(s, param++, 1.0 );
  }
  if( resYMulti.size() > 0){
    gsl_vector_set(s, param++, 1.0 );
  }
  if( resXYMulti.size() > 0){
    gsl_vector_set(s, param++, 1.0 );
  }

  //Alignment
  for(size_t ii = 0; ii < xShiftIndex.size(); ii++){
    gsl_vector_set(s, param++, 50.0);
  }
  for(size_t ii = 0; ii < yShiftIndex.size(); ii++){
    gsl_vector_set(s, param++, 50.0);
  }
  for(size_t ii = 0; ii < xScaleIndex.size(); ii++){
    gsl_vector_set(s, param++, 0.2);
  }
  for(size_t ii = 0; ii < yScaleIndex.size(); ii++){
    gsl_vector_set(s, param++, 0.2);
  }
  for(size_t ii = 0; ii < zRotIndex.size(); ii++){
    gsl_vector_set(s, param++, 0.2);
  }
  for(size_t ii = 0; ii < zPosIndex.size(); ii++){
    gsl_vector_set(s, param++, 10000.0);
  }
  return(s);
}

double estimateSimplex(const gsl_vector* v, void * params){
  //A wrapper function for the simplex search
  Minimizer* minimize = (Minimizer*) params;
  EstMat& estMat = minimize->mat;
  estMat.estToSystem(v);
  int zPrev = -999999999;
  for(size_t ii = 0; ii < estMat.system.planes.size(); ii++){
    if(zPrev > estMat.system.planes.at(ii).getZpos()){
      cout << "Bad z poses! " << ii << endl;
      //return( GSL_NAN );
    }
    zPrev = estMat.system.planes.at(ii).getZpos();
  }
  return((*minimize)());
}

void EstMat::simplexSearch(Minimizer* minimizeMe){
  //Do the simplex search on the configured Minimizer object
  cout << "Initial guesses" << endl;
  printParams( (char*) "params[\"RadiationLengths\"]", radLengths, radLengthsIndex.size() or radLengthsMulti.size() );
  bool resXYp = resXYIndex.size() or resXYMulti.size();
  printParams( (char*) "params[\"ResolutionX\"]", resX, resXIndex.size() or resXMulti.size() or resXYp);
  printParams( (char*) "params[\"ResolutionY\"]", resY, resYIndex.size() or resYMulti.size() or resXYp);
  printParams( (char*) "params[\"ZPosition\"]", zPos, zPosIndex.size());
  
  itMax = tracks.size();

  for(int ii = 0; ii < 3; ii++){
    size_t nSteps = 200;
    if( ii == 2){ nSteps = 600; }
    const gsl_multimin_fminimizer_type *T = gsl_multimin_fminimizer_nmsimplex2;
    gsl_multimin_fminimizer *s = NULL;
    gsl_vector *ss, *x;
    gsl_multimin_function minex_func;
    
    size_t iter = 0;
    int status;
    double size;
    size_t nParams = getNSimplexParams();
    
    /* Starting point */
    x = systemToEst();
    
    /* Set initial step sizes  */
    ss = simplesStepSize();
    
    /* Initialize method and iterate */
    minex_func.n = nParams;
    minex_func.f = estimateSimplex;
    minex_func.params = minimizeMe;
    
    s = gsl_multimin_fminimizer_alloc (T, nParams);
    gsl_multimin_fminimizer_set (s, &minex_func, x, ss);
    
    do{
      iter++;
      status = gsl_multimin_fminimizer_iterate(s);
      
      if (status){ break;}
      
      size = gsl_multimin_fminimizer_size (s);
      status = gsl_multimin_test_size (size, 1e-12);
      
      if (status == GSL_SUCCESS) {
	printf ("converged to minimum at\n");
      }

      if(iter % 10 == 0){
	cout << "Iteration " << iter << endl;
	printf ("%5d %10.3e %10.3e f() = %7.3f size = %.7f\n", 
		(int)iter,
		gsl_vector_get (s->x, 0), 
		gsl_vector_get (s->x, 1), 
		s->fval, size);
	printParams( (char*) "params[\"RadiationLengths\"]", radLengths, radLengthsIndex.size() or radLengthsMulti.size() );
	printParams( (char*) "params[\"ResolutionX\"]", resX, resXIndex.size() or resXMulti.size() or resXYp);
	printParams( (char*) "params[\"ResolutionY\"]", resY, resYIndex.size() or resYMulti.size() or resXYp);
	printParams( (char*) "params[\"XShift\"]", xShift, xShiftIndex.size());
	printParams( (char*) "params[\"YShift\"]", yShift, yShiftIndex.size());
	printParams( (char*) "params[\"XScale\"]", xScale, xScaleIndex.size());
	printParams( (char*) "params[\"YScale\"]", yScale, yScaleIndex.size());
	printParams( (char*) "params[\"ZRot\"]", zRot, zRotIndex.size());
	printParams( (char*) "params[\"ZPosition\"]", zPos, zPosIndex.size());
      }
    }
#if defined(USE_GEAR) //Called from Marlin
    while (status == GSL_CONTINUE && iter < 10000);
    break;
#else  //Called from sim
    while (status == GSL_CONTINUE && iter < nSteps);
#endif
    gsl_vector_free(x);
    gsl_vector_free(ss);
    gsl_multimin_fminimizer_free (s);
    cout << "Status: " << status << endl;
  }
}

void EstMat::quasiNewtonHomeMade(FwBw* minimizeMe){
  // Successive steps in Newtons method to find the minimum
  cout << "Initial guesses" << endl;
  printParams( (char*) "params[\"RadiationLengths\"]", radLengths, radLengthsIndex.size() or radLengthsMulti.size() );
  bool resXYp = resXYIndex.size() or resXYMulti.size();
  printParams( (char*) "params[\"ResolutionX\"]", resX, resXIndex.size() or resXMulti.size() or resXYp);
  printParams( (char*) "params[\"ResolutionY\"]", resY, resYIndex.size() or resYMulti.size() or resXYp);
  printParams( (char*) "params[\"ZPosition\"]", zPos, zPosIndex.size());

  size_t nParams = getNSimplexParams();
  gsl_vector* vc = systemToEst();
  itMax = tracks.size();
  double mseval = 0, fwbwval = 0;

  size_t resSize = resXIndex.size() + resYIndex.size();
  cout << "resSize " << resSize << endl;

  for(int iteration  = 0; iteration < 40; iteration++){
    cout << "it " << iteration << endl;
    for(size_t ii = 0; ii < nParams; ii++){
      bool doMse = (ii < resSize);// and (iteration > 5); 

      double val = gsl_vector_get(vc, ii);
      double step = val * .04 + 0.0001; //1 percent fails for floats

      //5 point stencil to get 1 and second detivative of param ii
      estToSystem(vc);
      double p0h = (*minimizeMe)();
      fwbwval = p0h;
      mseval = minimizeMe->retVal2;

      if(doMse) { 
	p0h = minimizeMe->retVal2; 
      }
      
      gsl_vector_set(vc, ii, val + step);
      estToSystem(vc);
      double p1h = (*minimizeMe)();
      if(doMse) { p1h = minimizeMe->retVal2; }
      
      gsl_vector_set(vc, ii, val + 2.0 * step);
      estToSystem(vc);
      double p2h = (*minimizeMe)();
      if(doMse) { p2h = minimizeMe->retVal2; }

      gsl_vector_set(vc, ii, val - step);
      estToSystem(vc);
      double m1h = (*minimizeMe)();
      if(doMse) { m1h = minimizeMe->retVal2; }
      
      gsl_vector_set(vc, ii, val - 2.0 * step);
      estToSystem(vc);
      double m2h = (*minimizeMe)();
      if(doMse) { m2h = minimizeMe->retVal2; }
      
      gsl_vector_set(vc, ii, val);
      estToSystem(vc);

      double firstDeriv = ( - 1.0 * p2h + 8.0 * p1h - 8.0 * m1h + 1.0 * m2h) / (12.0 * step);
      double secondDeriv = ( -1.0 * p2h + 16 * p1h - 30.0 * p0h + 16.0 * m1h - 1.0 * m2h)/( 12.0 * step * step );
      double newVal = firstDeriv / secondDeriv;
      //cout << firstDeriv << " , " << secondDeriv << " , " << val << " , " << newVal << endl;
      
      //Did we eben get a number?
      if(isnan(newVal) or isinf(newVal)){
	cout << "Nan step "<< endl;
	gsl_vector_set(vc, ii, val );
	estToSystem(vc);
      } else {
	if(secondDeriv < 0 ){ 
	  iteration = 0;
	  cout << "Maximizing, " << val << " , " << val - newVal << endl;
	  if( fabs(newVal) > fabs( val ) * 0.1 ){//Daming
	    //newVal = 0.25 * val;
	    newVal = 0.15 * val;
	    if(firstDeriv < 0){
	      newVal *= -1.0;
	    }
	  } else {
	    newVal *= -1.0;
	  }
	}
	
	while(fabs(newVal) > 5 * fabs(val) + 0.00001){
	  cout << "Large step, damping: " << val << " , " << val - newVal << endl;
	  newVal *= 0.1;
	}
	gsl_vector_set(vc, ii, fabs(val - newVal) );
	estToSystem(vc);
      }
    }
    
    printf ("fwbw() = %10.5f mse() = %10.5f\n", fwbwval, mseval);
    printParams( (char*) "params[\"RadiationLengths\"]", radLengths, radLengthsIndex.size() or radLengthsMulti.size() );
    printParams( (char*) "params[\"ResolutionX\"]", resX, resXIndex.size() or resXMulti.size() or resXYp);
    printParams( (char*) "params[\"ResolutionY\"]", resY, resYIndex.size() or resYMulti.size() or resXYp);
    printParams( (char*) "params[\"XShift\"]", xShift, xShiftIndex.size());
    printParams( (char*) "params[\"YShift\"]", yShift, yShiftIndex.size());
    printParams( (char*) "params[\"XScale\"]", xScale, xScaleIndex.size());
    printParams( (char*) "params[\"YScale\"]", yScale, yScaleIndex.size());
    printParams( (char*) "params[\"ZRot\"]", zRot, zRotIndex.size());
    printParams( (char*) "params[\"ZPosition\"]", zPos, zPosIndex.size());
  }
  gsl_vector_free(vc);
}
