/*
  This is a two-dimensional data-parallel implementation of the Kalman filter in the
  information filter formulation.

  The 2d fitter can be used twice to fit 4d data, as long as the 4x4 covariance matrix is
  block diagonal.

  The parameters are x = [x, dx]
  The covariance matrix is
  C = |xx  xdx |
      |xdx dxdx|
  incormation filter stores information matrix W = inv(C) and information vector i = Wx
*/

typedef struct{
  float4 x;
  float4 dx;
  //
  float4 xx;
  float4 xdx;
  float4 dxdx;
} estimate;


static inline void update(estimate* e, float4 meas, float invVar){
  e->x += invVar * meas;
  e->xx += invVar;
}

static inline void predict(estimate* e, float dz){
  //Propagate as a straight line a distance dz. 
  //Linear error propagation of information matrix.
  //F'WF, where F = [1 -dz][0 1]

  float4 xdxtmp = e->xdx;
  
  e->xdx  += dz * e->xx;
  e->dxdx += dz * xdxtmp + dz * e->xdx;
  
  //Update of information vector i = Fi
  e->dx += dz * e->x;
}

static inline void addScatter(estimate* e, float invScatterVar){
  //Woodbury matrix identity. Assume diagonal scattering matrix, and sparse covariance, and
  //diagonalmapping matrices.
  
  float4 scatterx = 1.0f/(e->dxdx + invScatterVar);
  
  //Update info vector
  e->x  -= scatterx * e->xdx  * e->dx;
  e->dx -= scatterx * e->dxdx * e->dx;
  
  //Update info matrix
  e->xx   -= e->xdx  * e->xdx  * scatterx;
  e->xdx  -= e->dxdx * e->xdx  * scatterx;
  e->dxdx -= e->dxdx * e->dxdx * scatterx;
}

static inline float getChi2Inc(estimate* e,
			       float4 meas,
			       float var){
  //We need the explicit parameters, not the information stuff. Invert and solve.
  //Get partial inverted matrix, do not need dxdx
  float4 det = 1.0f/(e->xx * e->dxdx - e->xdx * e->xdx);
  float4 myXx = det * e->dxdx;
  float4 myXdx = det * -e->xdx;
  //Get explicit x, do not need dx
  float4 myx = myXx * e->x + myXdx * e->dx;
  
  //Get residuals and 
  float4 resid = myx - meas;
  float4 chi2 = resid*resid/(myXx + var);
  //chi2 increment
  return( (chi2.x + chi2.y + chi2.z + chi2.w) / 4.0f);
  //return(myx);
}

//Create a bunch of near clone kernels to avoid branching. Cases are first measurement plane,
//second measurement plane and all other planes. All three are for forward and backward
//running filters.

//Normal planes need a prediction, the estimated chi2inc, an update, and added scattering
//Measurement is on front side, so update then scatter.
__kernel void processNormalPlaneFW(__global float4* measX, __global float4* measY,
				   __global float4* x, __global float4* dx,
				   __global float4* xx, __global float4* dxdx,
				   __global float4* xdx,
				   __global float4* y, __global float4* dy,
				   __global float4* yy, __global float4* dydy,
				   __global float4* ydy, 
				   __global float* chi2x, __global float* chi2y,
				   __private float varX, __private float varY, 
				   __private float invScatterVar, __private float dz){
  int gid = get_global_id(0);
  //Start with parameters x, dx.
  estimate e = {x[gid], dx[gid], xx[gid], xdx[gid], dxdx[gid]};
  float4 meas = measX[gid];
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

//The first plane has no predict, no chi2 increment
__kernel void processFirstPlaneFW(__global float4* measX, __global float4* measY,
				 __global float4* x, __global float4* dx,
				 __global float4* xx, __global float4* dxdx,
				 __global float4* xdx,
				 __global float4* y, __global float4* dy,
				 __global float4* yy, __global float4* dydy,
				 __global float4* ydy, 
				 __global float* chi2x, __global float* chi2y,
				 __private float varX, __private float varY, 
				 __private float invScatterVar, __private float dz){
  int gid = get_global_id(0);
  //Start with parameters x, dx. Set them all to 0;
  //estimate e = {x[gid], dx[gid], xx[gid], xdx[gid], dxdx[gid]};
  estimate e = {(float4) {0.0f, 0.0f, 0.0f, 0.0f},
		(float4) {0.0f, 0.0f, 0.0f, 0.0f},
		(float4) {0.0f, 0.0f, 0.0f, 0.0f},
		(float4) {0.0f, 0.0f, 0.0f, 0.0f},
		(float4) {0.0f, 0.0f, 0.0f, 0.0f}};
  float4 meas = measX[gid];
  update(&e,meas, 1.0f/varX);
  addScatter(&e, invScatterVar);
  x[gid]    = e.x;  dx[gid]   = e.dx;
  xx[gid]   = e.xx;  xdx[gid]  = e.xdx;  dxdx[gid] = e.dxdx;
  
  //Then do y, dy
  //e.x = y[gid];   e.dx = dy[gid];  
  //e.xx = yy[gid];  e.xdx = ydy[gid];  e.dxdx = dydy[gid];  
  e.x = (float4) {0.0f, 0.0f, 0.0f, 0.0f};
  e.dx = (float4) {0.0f, 0.0f, 0.0f, 0.0f};
  e.xx = (float4) {0.0f, 0.0f, 0.0f, 0.0f};
  e.xdx = (float4) {0.0f, 0.0f, 0.0f, 0.0f};
  e.dxdx = (float4) {0.0f, 0.0f, 0.0f, 0.0f};
  meas = measY[gid];
  update(&e,meas, 1.0f/varY);
  addScatter(&e, invScatterVar);
  y[gid]    = e.x;   dy[gid]   = e.dx;
  yy[gid]   = e.xx;   ydy[gid]  = e.xdx;  dydy[gid] = e.dxdx;
}

//The second plane has no chi2 increment
__kernel void processSecondPlaneFW(__global float4* measX, __global float4* measY,
				   __global float4* x, __global float4* dx,
				   __global float4* xx, __global float4* dxdx,
				   __global float4* xdx,
				   __global float4* y, __global float4* dy,
				   __global float4* yy, __global float4* dydy,
				   __global float4* ydy, 
				   __global float* chi2x, __global float* chi2y,
				   __private float varX, __private float varY, 
				   __private float invScatterVar, __private float dz){
  int gid = get_global_id(0);
  //Start with parameters x, dx.
  estimate e = {x[gid], dx[gid], xx[gid], xdx[gid], dxdx[gid]};
  float4 meas = measX[gid];
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


//Backward running kernels. 

//Normal planes need a prediction, scatterm the estimated chi2inc, and an update
//Measurement is on front side, so scatter then update.
__kernel void processNormalPlaneBW(__global float4* measX, __global float4* measY,
				   __global float4* x, __global float4* dx,
				   __global float4* xx, __global float4* dxdx,
				   __global float4* xdx,
				   __global float4* y, __global float4* dy,
				   __global float4* yy, __global float4* dydy,
				   __global float4* ydy, 
				   __global float* chi2x, __global float* chi2y,
				   __private float varX, __private float varY, 
				   __private float invScatterVar, __private float dz){
  int gid = get_global_id(0);
  //Start with parameters x, dx.
  estimate e = {x[gid], dx[gid], xx[gid], xdx[gid], dxdx[gid]};
  float4 meas = measX[gid];
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

//The first plane has no predict, no chi2 increment, no scatter
__kernel void processFirstPlaneBW(__global float4* measX, __global float4* measY,
				  __global float4* x, __global float4* dx,
				  __global float4* xx, __global float4* dxdx,
				  __global float4* xdx,
				  __global float4* y, __global float4* dy,
				  __global float4* yy, __global float4* dydy,
				  __global float4* ydy, 
				  __global float* chi2x, __global float* chi2y,
				  __private float varX, __private float varY, 
				  __private float invScatterVar, __private float dz){
  int gid = get_global_id(0);
  //Start with parameters x, dx.
  estimate e = {(float4) {0.0f, 0.0f, 0.0f, 0.0f},
		(float4) {0.0f, 0.0f, 0.0f, 0.0f},
		(float4) {0.0f, 0.0f, 0.0f, 0.0f},
		(float4) {0.0f, 0.0f, 0.0f, 0.0f},
		(float4) {0.0f, 0.0f, 0.0f, 0.0f}};

  float4 meas = measX[gid];
  update(&e,meas, 1.0f/varX);
  x[gid]    = e.x;  dx[gid]   = e.dx;
  xx[gid]   = e.xx;  xdx[gid]  = e.xdx;  dxdx[gid] = e.dxdx;
  
  //Then do y, dy
  e.x = (float4) {0.0f, 0.0f, 0.0f, 0.0f};
  e.dx = (float4) {0.0f, 0.0f, 0.0f, 0.0f};
  e.xx = (float4) {0.0f, 0.0f, 0.0f, 0.0f};
  e.xdx = (float4) {0.0f, 0.0f, 0.0f, 0.0f};
  e.dxdx = (float4) {0.0f, 0.0f, 0.0f, 0.0f};

  meas = measY[gid];
  update(&e,meas, 1.0f/varY);
  y[gid]    = e.x;   dy[gid]   = e.dx;
  yy[gid]   = e.xx;   ydy[gid]  = e.xdx;  dydy[gid] = e.dxdx;
}

//The second plane has no chi2 increment
__kernel void processSecondPlaneBW(__global float4* measX, __global float4* measY,
				   __global float4* x, __global float4* dx,
				   __global float4* xx, __global float4* dxdx,
				   __global float4* xdx,
				   __global float4* y, __global float4* dy,
				   __global float4* yy, __global float4* dydy,
				   __global float4* ydy, 
				   __global float* chi2x, __global float* chi2y,
				   __private float varX, __private float varY, 
				   __private float invScatterVar, __private float dz){
  int gid = get_global_id(0);
  //Start with parameters x, dx.
  estimate e = {x[gid], dx[gid], xx[gid], xdx[gid], dxdx[gid]};
  float4 meas = measX[gid];
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
