/*
  This is a two-dimensional data-parallel implementation of the Kalman filter in the
  information filter formulation.

  The 2d fitter is used twice to fit 4d data with a 4x4 covariance matrix that is block
  diagonal.

  The parameters are x = [x, dx]
  The covariance matrix is
  C = |xx  xdx |
      |xdx dxdx|
  incormation filter stores information matrix W = inv(C) and information vector i = Wx

  float8s are used to do partial reduction of the chi2 vectors, and to limit memory bandwith
  when reading it back.
*/

typedef struct{
  //Informationvector
  float8 x;
  float8 dx;
  //Symmetrix information matrix.
  float8 xx;
  float8 xdx;
  float8 dxdx;
} estimate;


static inline void update(estimate* e, float8 meas, float invVar){
  e->x += invVar * meas;
  e->xx += invVar;
}

static inline void predict(estimate* e, float dz){
  //Propagate as a straight line a distance dz. 
  //Linear error propagation of information matrix.
  //F'WF, where F = [1 -dz][0 1]

  float8 xdxtmp = e->xdx;
  
  e->xdx  += dz * e->xx;
  e->dxdx += dz * xdxtmp + dz * e->xdx;
  
  //Update of information vector i = Fi
  e->dx += dz * e->x;
}

static inline void addScatter(estimate* e, float invScatterVar){
  //Woodbury matrix identity. Assume diagonal scattering matrix, and sparse covariance, and
  //diagonalmapping matrices.
  
  float8 scatterx = 1.0f/(e->dxdx + invScatterVar);
  
  //Update info vector
  e->x  -= scatterx * e->xdx  * e->dx;
  e->dx -= scatterx * e->dxdx * e->dx;
  
  //Update info matrix
  e->xx   -= e->xdx  * e->xdx  * scatterx;
  e->xdx  -= e->dxdx * e->xdx  * scatterx;
  e->dxdx -= e->dxdx * e->dxdx * scatterx;
}

static inline float getChi2Inc(estimate* e,
			       float8 meas,
			       float var){
  //We need the explicit parameters, not the information stuff. Invert and solve.
  //Get partial inverted matrix, do not need dxdx
  float8 det = 1.0f/(e->xx * e->dxdx - e->xdx * e->xdx);
  float8 myXx = det * e->dxdx;
  float8 myXdx = det * -e->xdx;
  //Get explicit x, do not need dx
  float8 myx = myXx * e->x + myXdx * e->dx;
  
  //Get residuals and 
  float8 resid = myx - meas;
  float8 chi2 = resid*resid/(myXx + var);
  //chi2 increment
  return( (chi2.s0 + chi2.s1 + chi2.s2 + chi2.s3 + 
	   chi2.s4 + chi2.s5 + chi2.s6 + chi2.s7) / 8.0f);
  //return(myx);
}

/*
Create a bunch of near clone kernels to avoid branching. The kernels all update the
parameter estimates with a new measurement. If possible, the chi2 increments are
calculated.

Cases are first measurement plane, second measurement plane and all other planes. All three
cases are different for forward and backward running filters. Forward needs update, then
scatter, backward needs scatter then update.
*/

//The first plane has no predict, no chi2 increment. The parameter estimates are set to 0.0f.
__kernel void processFirstPlaneFW(__global float8* measX, __global float8* measY,
				 __global float8* x, __global float8* dx,
				 __global float8* xx, __global float8* dxdx,
				 __global float8* xdx,
				 __global float8* y, __global float8* dy,
				 __global float8* yy, __global float8* dydy,
				 __global float8* ydy, 
				 __global float* chi2x, __global float* chi2y,
				 __private float varX, __private float varY, 
				 __private float invScatterVar, __private float dz){
  int gid = get_global_id(0);
  //Start with parameters x, dx. Set them all to 0 (float8);
  estimate e = {(float8) {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
		(float8) {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
		(float8) {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
		(float8) {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
		(float8) {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}};
  float8 meas = measX[gid];
  update(&e,meas, 1.0f/varX);
  addScatter(&e, invScatterVar);
  x[gid]    = e.x;  dx[gid]   = e.dx;
  xx[gid]   = e.xx;  xdx[gid]  = e.xdx;  dxdx[gid] = e.dxdx;
  
  //Then do y, dy
  e.x = (float8) {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  e.dx = (float8) {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  e.xx = (float8) {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  e.xdx = (float8) {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  e.dxdx = (float8) {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  meas = measY[gid];
  update(&e,meas, 1.0f/varY);
  addScatter(&e, invScatterVar);
  y[gid]    = e.x;   dy[gid]   = e.dx;
  yy[gid]   = e.xx;   ydy[gid]  = e.xdx;  dydy[gid] = e.dxdx;
}

//The second plane has no chi2 increment
__kernel void processSecondPlaneFW(__global float8* measX, __global float8* measY,
				   __global float8* x, __global float8* dx,
				   __global float8* xx, __global float8* dxdx,
				   __global float8* xdx,
				   __global float8* y, __global float8* dy,
				   __global float8* yy, __global float8* dydy,
				   __global float8* ydy, 
				   __global float* chi2x, __global float* chi2y,
				   __private float varX, __private float varY, 
				   __private float invScatterVar, __private float dz){
  int gid = get_global_id(0);
  //Start with parameters x, dx.
  estimate e = {x[gid], dx[gid], xx[gid], xdx[gid], dxdx[gid]};
  float8 meas = measX[gid];
  predict(&e,dz);
  update(&e,meas, 1.0f/varX);
  addScatter(&e, invScatterVar);
  x[gid]    = e.x;   dx[gid]   = e.dx;
  xx[gid]   = e.xx;   xdx[gid]  = e.xdx;   dxdx[gid] = e.dxdx;
  
  //Then do y, dy
  e.x = y[gid];   e.dx = dy[gid];  
  e.xx = yy[gid];  e.xdx = ydy[gid];  e.dxdx = dydy[gid];  
  meas = measY[gid];
  predict(&e,dz);
  update(&e,meas, 1.0f/varY);
  addScatter(&e, invScatterVar);
  y[gid]    = e.x;   dy[gid]   = e.dx;
  yy[gid]   = e.xx;   ydy[gid]  = e.xdx;   dydy[gid] = e.dxdx;
}

//Normal planes need a prediction, the estimated chi2inc, an update, and added scattering
__kernel void processNormalPlaneFW(__global float8* measX, __global float8* measY,
				   __global float8* x, __global float8* dx,
				   __global float8* xx, __global float8* dxdx,
				   __global float8* xdx,
				   __global float8* y, __global float8* dy,
				   __global float8* yy, __global float8* dydy,
				   __global float8* ydy, 
				   __global float* chi2x, __global float* chi2y,
				   __private float varX, __private float varY, 
				   __private float invScatterVar, __private float dz){
  int gid = get_global_id(0);
  //Start with parameters x, dx.
  estimate e = {x[gid], dx[gid], xx[gid], xdx[gid], dxdx[gid]};
  float8 meas = measX[gid];
  predict(&e,dz);
  chi2x[gid] = getChi2Inc(&e, meas, varX);
  update(&e,meas, 1.0f/varX);
  addScatter(&e, invScatterVar);
  //Copy back to global
  x[gid]    = e.x;  dx[gid]   = e.dx;
  xx[gid]   = e.xx;   xdx[gid]  = e.xdx;   dxdx[gid] = e.dxdx;
  
  //Then do y, dy
  e.x = y[gid];   e.dx = dy[gid];  
  e.xx = yy[gid];  e.xdx = ydy[gid];  e.dxdx = dydy[gid];  

  meas = measY[gid];
  predict(&e,dz);
  chi2y[gid] = getChi2Inc(&e, meas, varY);
  update(&e,meas, 1.0f/varY);
  addScatter(&e, invScatterVar);
  //Copy back to global
  y[gid]    = e.x; dy[gid]   = e.dx;
  yy[gid]   = e.xx; ydy[gid]  = e.xdx; dydy[gid] = e.dxdx;
}



//Backward running kernels. 
//The first plane has no predict, no chi2 increment, no scatter
__kernel void processFirstPlaneBW(__global float8* measX, __global float8* measY,
				  __global float8* x, __global float8* dx,
				  __global float8* xx, __global float8* dxdx,
				  __global float8* xdx,
				  __global float8* y, __global float8* dy,
				  __global float8* yy, __global float8* dydy,
				  __global float8* ydy, 
				  __global float* chi2x, __global float* chi2y,
				  __private float varX, __private float varY, 
				  __private float invScatterVar, __private float dz){
  int gid = get_global_id(0);
  //Start with parameters x, dx.
  estimate e = {(float8) {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
		(float8) {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
		(float8) {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
		(float8) {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
		(float8) {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}};

  float8 meas = measX[gid];
  update(&e,meas, 1.0f/varX);
  x[gid]    = e.x;  dx[gid]   = e.dx;
  xx[gid]   = e.xx;  xdx[gid]  = e.xdx;  dxdx[gid] = e.dxdx;
  
  //Then do y, dy
  e.x = (float8) {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  e.dx = (float8) {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  e.xx = (float8) {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  e.xdx = (float8) {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  e.dxdx = (float8) {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

  meas = measY[gid];
  update(&e,meas, 1.0f/varY);
  y[gid]    = e.x;   dy[gid]   = e.dx;
  yy[gid]   = e.xx;   ydy[gid]  = e.xdx;  dydy[gid] = e.dxdx;
}

//The second plane has no chi2 increment
__kernel void processSecondPlaneBW(__global float8* measX, __global float8* measY,
				   __global float8* x, __global float8* dx,
				   __global float8* xx, __global float8* dxdx,
				   __global float8* xdx,
				   __global float8* y, __global float8* dy,
				   __global float8* yy, __global float8* dydy,
				   __global float8* ydy, 
				   __global float* chi2x, __global float* chi2y,
				   __private float varX, __private float varY, 
				   __private float invScatterVar, __private float dz){
  int gid = get_global_id(0);
  //Start with parameters x, dx.
  estimate e = {x[gid], dx[gid], xx[gid], xdx[gid], dxdx[gid]};
  float8 meas = measX[gid];
  predict(&e,dz);
  addScatter(&e, invScatterVar);
  update(&e,meas, 1.0f/varX);
  x[gid]    = e.x;   dx[gid]   = e.dx;
  xx[gid]   = e.xx;   xdx[gid]  = e.xdx;   dxdx[gid] = e.dxdx;
  
  //Then do y, dy
  e.x = y[gid];   e.dx = dy[gid];  
  e.xx = yy[gid];  e.xdx = ydy[gid];  e.dxdx = dydy[gid];  
  meas = measY[gid];
  predict(&e,dz);
  addScatter(&e, invScatterVar);
  update(&e,meas, 1.0f/varY);
  y[gid]    = e.x;   dy[gid]   = e.dx;
  yy[gid]   = e.xx;   ydy[gid]  = e.xdx;   dydy[gid] = e.dxdx;
}

//Normal planes need a prediction, scatter,  the estimated chi2inc, and an update
__kernel void processNormalPlaneBW(__global float8* measX, __global float8* measY,
				   __global float8* x, __global float8* dx,
				   __global float8* xx, __global float8* dxdx,
				   __global float8* xdx,
				   __global float8* y, __global float8* dy,
				   __global float8* yy, __global float8* dydy,
				   __global float8* ydy, 
				   __global float* chi2x, __global float* chi2y,
				   __private float varX, __private float varY, 
				   __private float invScatterVar, __private float dz){
  int gid = get_global_id(0);
  //Start with parameters x, dx.
  estimate e = {x[gid], dx[gid], xx[gid], xdx[gid], dxdx[gid]};
  float8 meas = measX[gid];
  predict(&e,dz);
  addScatter(&e, invScatterVar);
  chi2x[gid] = getChi2Inc(&e, meas, varX);
  update(&e,meas, 1.0f/varX);
  //Copy back to global
  x[gid]    = e.x;  dx[gid]   = e.dx;
  xx[gid]   = e.xx;   xdx[gid]  = e.xdx;   dxdx[gid] = e.dxdx;
  
  //Then do y, dy
  e.x = y[gid];   e.dx = dy[gid];  
  e.xx = yy[gid];  e.xdx = ydy[gid];  e.dxdx = dydy[gid];  
  meas = measY[gid];
  predict(&e,dz);
  addScatter(&e, invScatterVar);
  chi2y[gid] = getChi2Inc(&e, meas, varY);
  update(&e,meas, 1.0f/varY);
  //Copy back to global
  y[gid]    = e.x; dy[gid]   = e.dx;
  yy[gid]   = e.xx; ydy[gid]  = e.xdx; dydy[gid] = e.dxdx;
}

