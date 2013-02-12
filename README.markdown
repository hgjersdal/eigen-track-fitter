#eigen2-track-fitter

A straight line track fitter for test beam analysis of pixel partickle detectors.

The implementation contains a Combinatorial Kalman Filter for track finding, as well as implementation of the standard formulation of the Kalman Filter, an information matrix implementation of the Kalman Filter, and the Deterministic annealing filter. 

The implementations are template classes, taking the fitter type (double or float), and the dimention of the fitter. The implementation currently only works for 4 parameters(straight line tracks). Due to optimizations using the code for 5 parameters would require many functions to be overloaded, not only in the propagation. 

Also contains an implementation of the methods for estimating material and resolutions of several detector planes simultaneously, as described in the paper "Optimizing track reconstruction by simultaneous estimation of material and resolutions" (http://dx.doi.org/10.1088/1748-0221/8/01/P01009). An application of the method on toy simulated data is also available.

Code for using the fitter and material estimator from the EUTELESCOPE software framework is also included.

##Dependencies

The track fitter(EUTelTrackerSystem, EUTelDafEigenFitter) depends on the linear algebra package eigen2. The material and resolution estimator relies on the GNU scientific library for minimization and boost_threads for threading if DOTHREADS is defined. Marlin does not include boost, so changing the build environment is needed if one wants threading.

The simulation experiments use root for plotting.

##Building and running the material estimator

	   make estmat
	   ./estmat <option>
where option is one of `fwbw`, `sdr1`, `sdr2`, `sdr3`, or `hybr`

This will simulate then estimate the resolution and material distribution of 100 track samples. This takes a while, and will run 4 threads.

## Caveats
Note that the material estimator does not work well for electrons. Electrons lose a substantial amount energy through Bremsstrahlung, a highly non Gaussian process. This means the beam energy after material has been traversed in non uniform, and not Gaussianly distributed. Since there is no way of estimating the particle energy from a straight line track, all beam particles must be assumed to have the same amount of scattering, and this is simply not a good approximation. A method that works somewhat, is to use the FWBW estimator on a track sample after a very tight cut on the chi2 (in the area of maybe n.d.o.f. times 2 or 3).

Another important thing to note is that the plane thickness is treated as uniform across an infinite detector plane. The geometry does not take into account that outside the sensor chip, there is another thickness or no material at all. This method should find the optimal scattering variance for tracks passing through ALL sensor chips, if the material across the active region is uniform. If a track passes outside the sensor chip, the amount of material is not correctly estimated. This should be fairly straight forward to improve with a more advanced geometry description. It might even work if the planes are non infinite in the fitter code, and one uses passive planes in GEAR.

The amount of material in the plane is not really what is being optimized, the amount of scattering that occurs in the plane is. This is presented in X/X_0, but the X/X_0 value will probably be somewhat biased, since energy loss of the particle is not taken into account by the fitter.

All telescope and DUT planes must be included in the tracks fed to the material estimator. 