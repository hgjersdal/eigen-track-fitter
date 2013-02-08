def runConfig(params):
    params["GearFile"]  = "gear_telescope_apix_cern_november2010_dummyplanesz.xml"
    params["HistoInfo"] = "histoinfo.xml"
    params["DUTPlanes"] = "10 11 12"
    params["BADPlanes"] = "30 31"
    params["TELPlanes"] = "0 1 2 3 4 5"
    params["TelAlignmentDb"] = "run21198-alignment-tel-db.slcio"
    params["DutAlignmentDb"] = "run21198-alignment-dut-db.slcio"
    params["NoiseDb"]        = "run21198-noise-db.slcio"
    params["DataSet"] = "[21198, 21199, 21200, 21201, 21202, 21203, 21204, 21205, 21207, 21208, 21209, 21210, 21211, 21212, 21213, 21215, 21216, 21217, 21218, 21219, 21220, 21221]"
    params["BeamEnergy"] = "120.0"
    params["ColMin"] = "2"
    params["ColMax"] = "16"
    params["RowMin"] = "8"
    params["RowMax"] = "146"
    
    params["XShift"] = "  0.000000 1.104416 1.813742 1.245093 -1.073568 -4.079643 1.125459 0.866977 0.000000 "
    params["YShift"] = "  0.000000 -0.971513 -1.918661 -0.301787 -4.855701 -20.751425 -1.507555 -0.679969 0.000000 "
    params["XScale"] = "  0.000000 -0.000404 -0.000739 -0.000893 -0.000485 -0.000652 -0.000568 -0.000358 0.000000 "
    params["YScale"] = "  0.000000 -0.000416 -0.000685 -0.000079 -0.001085 -0.011959 -0.000605 -0.000333 0.000000 "
    params["ZRot"] = "  0.000000 -0.000284 -0.000646 -0.000716 -0.000737 -0.001077 -0.001268 -0.001381 -0.001596 "

    params["RadiationLengths"] = "  0.007300 0.080084 0.123408 0.081735 0.000792 0.167096 0.051738 0.080069 0.073000 "
    params["ResolutionX"] = "  4.00000 3.922218 3.741869 12.942457 9.957403 13.107492 4.021868 4.024720 4.00000 "
    params["ResolutionY"] = "  4.00000 3.904416 3.819799 115.165054 111.278938 112.286247 4.279577 3.957012 4.00000 "

    params["ZPos"] = "  0.000000 150000.000000 300000.000000 392700.0 441700.0 501700.0 680000.000000 830000.000000 980000.000000 "

    params["RadLengthIndex"] ="1 2 3 4 5 6 7"
    params["ResXIndex"] =     "1 2 3 4 5 6 7"
    params["ResYIndex"] =     "1 2 3 4 5 6 7"

    #params["RadLengthIndex"] =""
    #params["ResXIndex"] = ""
    #params["ResYIndex"] = ""

    params["ShiftXIndex"] = "1 2 3 4 5 6 7"
    params["ShiftYIndex"] = "1 2 3 4 5 6 7"
    params["ScaleXIndex"] = "1 2 3 4 5 6 7"
    params["ScaleYIndex"] = "1 2 3 4 5 6 7"
    params["ZRotIndex"] = "1 2 3 4 5 6 7"
    params["ZPosIndex"] = " "

    params["ShiftXIndex"] = ""
    params["ShiftYIndex"] = ""
    params["ScaleXIndex"] = ""
    params["ScaleYIndex"] = ""
    params["ZRotIndex"] = ""
    params["ZPosIndex"] = ""

#def noise(params):
#    params["TemplateFile"] = "noise-tmp.xml"

def noise(params):
    params["TemplateFile"] = "noise-pysub-tmp.xml"
    params["TelOccupancyThresh"] = "0.001"

def rawtohit(params):
    params["TemplateFile"] = "rawtohit-pysub-tmp.xml"
    #params["TelOccupancyThresh"] = "0.001"
    #params["DUTOccupancyThresh"] = "0.001"

def aligntel(params):
    params["TemplateFile"] = "kalman-align-tel-tmp.xml"
    params["RecordNumber"] = "10000000"
    params["SkipNEvents"] = "0"
    params["RunPede"] = "True"
    params["UseResidualCuts"] = "True"
    params["ResidualXMin"] = " 120   -20  -360  -9999  -9999  -9999  -60   60   -50  -9999  -9999  -9999  -9999"
    params["ResidualXMax"] = " 230    80  -260   9999   9999   9999   60  130    60   9999   9999   9999   9999"
    params["ResidualYMin"] = "  60  -180  -160  -9999  -9999  -9999  -80   90  -120  -9999  -9999  -9999  -9999"
    params["ResidualYMax"] = " 220     0   -50   9999   9999   9999   80  160    10   9999   9999   9999   9999"
    params["TelescopeResolution"] = "10 10 10 10000 10000 10000 10 10 10 10000 10000 10000 10000"
    params["DistanceMax"] = "125.0"
    params["MaxChi2"] = "2700"
    params["MinDxDz"] = "-0.0009"
    params["MaxDxDz"] = "0.0001"
    params["MinDyDz"] = "-0.0004" 
    params["MaxDyDz"] = "0.0005"
    params["ExcludePlanes"] = params["DUTPlanes"] + " " + params["BADPlanes"]
    params["FixedPlanes"] = "0"
    params["FixedTranslations"] = "4"
    params["FixedScales"] = "2"
    params["FixedZRotations"] = ""

def aligndut(params):
    params["TemplateFile"] = "daf-align-dut.xml"
    params["RecordNumber"] = "1000000"
    params["SkipNEvents"] = "0"
    params["MakePlots"] = "True"
    params["TelescopePlanes"] = params["TELPlanes"]
    params["DutPlanes"] = params["DUTPlanes"]
    params["TelResolution"] = "4.3"
    params["DutResolutionX"] = "144"
    params["DutResolutionY"] = "1165"
    params["FinderRadius"] = "300.0"
    params["Chi2Cutoff"] = "20.0"
    params["MaxChi2OverNdof"] = "4.5"
    params["RequireNTelPlanes"] = "6.0"
    params["NominalDxdz"] = "-0.00026"
    params["NominalDydz"] = "0.000"
    params["ScaleScatter"] = "1.00"
    params["ResidualXMin"] = "  1250   575   4800"
    params["ResidualXMax"] = "  1425   820   4980"
    params["ResidualYMin"] = " -1100  -300   -600"
    params["ResidualYMax"] = " -420    300     00"
    params["Translate"] = params["DUTPlanes"] + " 1 2 3 4 "
    params["TranslateZ"] = ""#params["DUTPlanes"]
    params["ZRotate"] = params["DUTPlanes"] + " 1 2 3 4 "
    params["Scale"] = params["DUTPlanes"] + " 1 2 3 4 "
    params["ScaleY"] = ""#params["DUTPlanes"]
    params["ScaleX"] = ""# params["DUTPlanes"]
    params["NDutHits"] = "3"

def daffitter(params):
    params["TemplateFile"] = "daf-fitter-tmp.xml"
    params["RecordNumber"] = "1000000"
    params["SkipNEvents"] = "0"
    params["MakePlots"] = "True"
    params["FitDuts"] = "False"
    params["AddToLCIO"] = "True"
    params["TelescopePlanes"] = params["TELPlanes"]
    params["DutPlanes"] = params["DUTPlanes"]
    params["TelResolution"] = "4.3"
    params["DutResolutionX"] = "7.2"
    params["DutResolutionY"] = "58.0 "
    params["FinderRadius"] = "300.0"
    params["Chi2Cutoff"] = "30.0"
    params["RequireNTelPlanes"] = "6.0"
    params["MaxChi2OverNdof"] = "15.0"
    params["NominalDxdz"] = "-0.0002"
    params["NominalDydz"] = "0.0001"
    params["ScaleScatter"] = "1.0"
    params["NDutHits"]= "0"

def matest(params):
    params["TemplateFile"] = "daf-matest-tmp.xml"
    params["RecordNumber"] = "10000000"
    params["SkipNEvents"] = "0"
    params["MakePlots"] = "True"
    params["TelescopePlanes"] = params["TELPlanes"]
    params["DutPlanes"] = params["DUTPlanes"]
    params["TelResolution"] = "4.3"
    params["DutResolutionX"] = "14.5"
    params["DutResolutionY"] = "115.0 "
    params["FinderRadius"] = "300.0"
    params["Chi2Cutoff"] = "20.0"
    params["RequireNTelPlanes"] = "6"
    params["MaxChi2OverNdof"] = "3.5"
    params["NominalDxdz"] = "-0.0002"
    params["NominalDydz"] = "0.0001"
    params["ScaleScatter"] = "1.0"
    params["NDutHits"]= "3"
    params["ColMin"] = "1 1 1"
    params["ColMax"] = "16 16 16"
    params["RowMin"] = "1 1 1"
    params["RowMax"] = "150 150 150"
    params["ResidualXMin"] = " -50    -50   -50"
    params["ResidualXMax"] = "  50     50    50"
    params["ResidualYMin"] = " -300  -300  -300"
    params["ResidualYMax"] = "  300   300   300"

# run Marlin
if __name__ == "__main__":
    from steeringGenerator import *
    functions = {"noise": noise,
                 "rawtohit": rawtohit,
                 "aligntel": aligntel,
                 "aligndut": aligndut,
                 "aligndut2": aligndut2,
                 "daffitter": daffitter,
                 "matest" : matest}
    jobMaker(functions, runConfig)
