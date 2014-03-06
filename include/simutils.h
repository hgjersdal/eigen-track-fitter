#ifndef SIMUTILS
#define SIMUTILS

// void printHisto(TH1D* histo){
//   cout << "(list" 
//        <<  " :min " << histo->GetXaxis()->GetXmin()
//        << " :bin-size " << histo->GetBinWidth( 0 ) << endl
//        << ":data (list" << endl;
//   for(int ii = 1; ii <= histo->GetNbinsX(); ii++){
//     cout << " " << histo->GetBinContent( ii );
//   }
//   cout << endl << "))" << endl;
// }



inline double normRand(){
  return( (double) random() / (double) RAND_MAX);
}

inline void gaussRand(double& x1, double& x2){
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

inline double getScatterSigma(double eBeam, double radLength){
  radLength = fabs(radLength);
  double scatterTheta = 0.0136f/ eBeam * sqrt( radLength ) *  (1.0f + 0.038f * std::log(radLength) );
  return(scatterTheta);
}

#endif
