//! Author Havard Gjersdal <haavagj@fys.uio.no>
// -*- mode: c++; mode: auto-fill; mode: flyspell-prog; -*-
/*
 *   This source code is part of the Eutelescope package of Marlin.
 */

#ifndef EUTELDAFBASE_H
#define EUTELDAFBASE_H

// built only if GEAR is available
#ifdef USE_GEAR

// eutelescope includes
#include "EUTelDafTrackerSystem.h"
#include "EUTelAlignmentConstant.h"

// marlin includes ".h"
#include "marlin/Processor.h"

// gear includes <.h>
#include <gear/SiPlanesParameters.h>
#include <gear/SiPlanesLayerLayout.h>

// lcio includes <.h>
#include <EVENT/LCRunHeader.h>
#include <EVENT/LCEvent.h>
#include <IMPL/LCCollectionVec.h>
#include <IMPL/TrackerHitImpl.h>

// AIDA includes <.h>
#if defined(USE_AIDA) || defined(MARLIN_USE_AIDA)
#include <AIDA/IHistogram1D.h>
#include <AIDA/IHistogram2D.h>
#endif

// system includes <>
#include <string>
#include <fstream>
#include <vector>
#include <map>

namespace eutelescope {
  class EUTelDafBase : public marlin::Processor {
  public:
    // Marlin processor interface funtions
    //! Returns a new instance of EUTelDafFitter
    // virtual Processor * newProcessor() {
    //   return new EUTelDafFitter;
    // }
    //! Default constructor
    EUTelDafBase ();
    EUTelDafBase (std::string);
    //! Called at the job beginning.
    virtual void init ();
    //! Called for every run.
    virtual void processRunHeader (LCRunHeader * run);
    //! Called every event
    virtual void processEvent (LCEvent * evt);
    //! Called after data processing.
    virtual void end();
    bool defineSystemFromData();
    
  protected:
    std::ofstream trackstream;
    //! Input hit collection name
    std::vector<std::string > _hitCollectionName;
    std::vector<std::string > _alignColNames;
    std::vector<int> _nRef;
    bool _initializedSystem;
    LCCollection* _hitCollection;

    std::vector<int> _colMin, _colMax, _rowMin, _rowMax;
    std::map<int, std::pair<int,int> > _rowMinMax, _colMinMax;

    // List of sensor IDs identifying telescopes and duts
    std::vector<int > _telPlanes;
    std::vector<int > _dutPlanes;

    //! resolution of sensor planes
    float _telResX, _telResY, _dutResX, _dutResY;
    //! Nominal beam energy
    float _eBeam;
    
    //! Radius for track finder finder
    /*! 
     * Track finder works by projecting all hits into plane 0, assuming a beam parallel to
     * the z-axis, then running a cluster finder on these hits. This radius determines
     * whether a hit is included or not.
     */
    float _clusterRadius;

    //! Cutoff value for DAF
    /*!
     * This determines the maximum distance between a track and a measurement for the
     * measurement to be included in the fit.
     */
    float _chi2cutoff;
    float _nXdz, _nYdz;
    int _nDutHits;
   
    float _nSkipMax;
    float _ndofMin;
    //! maximum allowed chi2 /ndof for track to be accepted.
    float _maxChi2;
    float _scaleScatter;

    virtual void dafInit(){;}
    virtual void dafEvent(LCEvent * evt){;}
    virtual void dafEnd(){;}
    virtual void dafParams(){;}
    bool checkClusterRegion(lcio::TrackerHitImpl* hit, int iden); // { return(true);}
    

    size_t getPlaneIndex(float zPos);
    float getScatterThetaVar(float radLength);
    void readHitCollection(LCEvent* event);
    void bookHistos();
    void bookDetailedHistos();
    void fillPlots(daffitter::TrackCandidate<float, 4>* track);
    void fillDetailPlots(daffitter::TrackCandidate<float, 4>* track);
    bool checkTrack(daffitter::TrackCandidate<float, 4> * track);
    int checkInTime(daffitter::TrackCandidate<float, 4> * track);
    void printStats();
    //alignment stuff
    void gearRotate(size_t index, size_t gearIndex);
    Matrix<float, 3, 1> applyAlignment(EUTelAlignmentConstant* alignment, Matrix<float, 3, 1> point);
    void alignRotate(std::string collectionName, LCEvent* event);
    void getPlaneNorm(daffitter::FitPlane<float> & pl);

    daffitter::TrackerSystem<float, 4> _system;
    std::map<float, int> _zSort;
    std::vector<float> _radLength, _xShift, _yShift, _xScale, _yScale, _zRot, _zPos;
    std::vector<float> _sigmaX, _sigmaY;

    //! Counters
    int _iRun, _iEvt, _nTracks, _nClusters, n_passedNdof, n_passedChi2OverNdof, n_passedIsnan;

    //! Silicon planes parameters as described in GEAR
    gear::SiPlanesParameters * _siPlanesParameters;
    gear::SiPlanesLayerLayout * _siPlanesLayerLayout;

#if defined(USE_AIDA) || defined(MARLIN_USE_AIDA)
    std::map<std::string, AIDA::IHistogram1D * > _aidaHistoMap;
    AIDA::IHistogram2D* _aidaZvHitX;
    AIDA::IHistogram2D* _aidaZvFitX;
    AIDA::IHistogram2D* _aidaZvHitY;
    AIDA::IHistogram2D* _aidaZvFitY;
#endif

    //! Fill histogram switch
    bool _histogramSwitch;
    //! LCIO switch
    bool _addToLCIO;
  };
}
#endif
#endif
