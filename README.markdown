#eigen-track-fitter

A straight line track fitter for test beam analysis of pixel partickle detectors.

The implementation contains a Combinatorial Kalman Filter for track finding, as well as implementation of the standard formulation of the Kalman Filter, an information matrix implementation of the Kalman Filter, and the Deterministic annealing filter. 

The implementations are template classes, taking the fitter type (double or float), and the dimention of the fitter. The implementation currently only works for 4 parameters(straight line tracks). Due to optimizations using the code for 5 parameters would require many functions to be overloaded, not only in the propagation. 

Also contains an implementation of the methods for estimating material and resolutions of several detector planes simultaneously, as described in the paper "Optimizing track reconstruction by simultaneous estimation of material and resolutions" (http://dx.doi.org/10.1088/1748-0221/8/01/P01009). An application of the method on toy simulated data is also available.

Code for using the fitter and material estimator from the EUTELESCOPE software framework is also included.

##Implementation

The core Kalman filter is implemented in `include/EUTelDafEigenFitter.tcc`.

Full track fitters, as well as pattern recognition algorithms are in `include/EUTelDafTrackerSystem.tcc` and `include/EUTelDafTrackerSystem.h`.

The material and resolution estimation methods are implemented in `include/estmat.h` and `src/estmat.cc`. An implementation of the "SDR2" method for material and resolution estimation has been made using openCL. The openCL kernel is in `include/openclkalmant5.cl`, the host code is in `src/sdr2clt3.cpp` and `include/sdr2clt3.h`.

The code can be interfaced from the EUTelescope framework with the classes implemented in `src/EUTelDaf*`. These interfaces might not be up to date, as I have not used the framework in a while.

`estmatapp.cc`, `noisesim.cc` and `simple.cc` set up and run simple simulation experiments for validating the implementations.

##Dependencies

The track fitter(EUTelTrackerSystem, EUTelDafEigenFitter) depends on the linear algebra package eigen3. The material and resolution estimator relies on the GNU scientific library for minimization and boost_threads for threading if DOTHREADS is defined. Marlin does not include boost, so changing the build environment is needed if one wants threading. One of the material and resolution estimators depend on openCL. 

The simulation experiments use root for plotting.

##Building and running

	   make simple
	   ./simple
will simulate 1M tracks, and refit them with CKF + DAF. Residuals and chi2 is plotted to plots/simple.root.

	   make estmat
	   ./estmat <option>
where option is one of `fwbw`, `sdr1`, `sdr2`, `sdr2cl`, `sdr3`, `hybr` or `align`.

This will simulate then estimate the resolution and material distribution of 100 track samples. This takes a while, and will run 4 threads. The `sdr2cl` method needs a GPU that can run the openCL kernel. The `sdr3` and `hybr` methods give the best estimates.

The `align` option works diffrently from all the others in that is uses a simplex search to minimize chi2 by manipulating alignment parameters, not thicknesses and resolution. The alignment parameters should all be 0. The default settings uses more iterations and restarts, due to the fact that there are more parameters to be estimated.

## Caveats
Note that the material estimator does not work well for electrons. Electrons lose a substantial amount energy through Bremsstrahlung, a highly non Gaussian process. This means the beam energy after material has been traversed in non uniform, and not Gaussianly distributed. Since there is no way of estimating the particle energy from a straight line track, all beam particles must be assumed to have the same amount of scattering, and this is simply not a good approximation. A method that works somewhat, is to use the FWBW estimator on a track sample after a very tight cut on the chi2 (in the area of maybe n.d.o.f. times 2 or 3).

Another important thing to note is that the plane thickness is treated as uniform across an infinite detector plane. The geometry does not take into account that outside the sensor chip, there is another thickness or no material at all. This method should find the optimal scattering variance for tracks passing through ALL sensor chips, if the material across the active region is uniform. If a track passes outside the sensor chip, the amount of material is not correctly estimated. This should be fairly straight forward to improve with a more advanced geometry description. It might even work if the planes are non infinite in the fitter code, and one uses passive planes in GEAR.

The amount of material in the plane is not really what is being optimized, the amount of scattering that occurs in the plane is. This is presented in X/X_0, but the X/X_0 value will probably be somewhat biased, since energy loss of the particle is not taken into account by the fitter.

All telescope and DUT planes must be included in the tracks fed to the material estimator. 