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

  measusement alternates between x and y, so float2, float4 or float 8 is needed.
*/

typedef struct{
  //Informationvector
  float4 x;
  float4 dx;
  //Symmetrix information matrix.
  float4 xx;
  float4 xdx;
  float4 dxdx;
} estimate;

static inline void update(estimate* e, float4 meas, float4 var){
  float4 invVar = 1.0f/var;
  e->x += invVar * meas;
  e->xx += invVar;
}

static inline void predict(estimate* e, float dz){
  //Propagate as a straight line a distance dz.
  //Linear error propagation of information matrix.
  //F'WF, where F = [1 -dz][0 1]

  float4 xdxdztmp = dz * e->xdx;
  
  e->xdx  += dz * e->xx;
  //e->dxdx += dz * xdxtmp + dz * e->xdx;
  e->dxdx += fma(dz, e->xdx, xdxdztmp);
  
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

static inline float4 getChi2Inc(estimate* e,
				float4 meas,
				float4 var){
  //We need the explicit parameters, not the information stuff. Invert and solve.
  //Get partial inverted matrix, do not need dxdx
  float4 det = 1.0f/(e->xx * e->dxdx - e->xdx * e->xdx);
  float4 myXx = det * e->dxdx;
  float4 myXdx = det * -e->xdx;
  //Get explicit x, do not need dx
  float4 myx = myXx * e->x + myXdx * e->dx;
  
  //Get residuals and
  float4 resid = myx - meas;
  //chi2 increment
  return( (resid*resid)/(myXx + var) );
}

static inline float4 fitPlaneFW(estimate* e, float4 meas, float2 mvar, float scatter, float dz){
  float4 var = mvar.s0101;
  predict(e,dz);
  float4 chi2 = getChi2Inc(e, meas, var);
  update(e,meas, var);
  addScatter(e, scatter);
  return(chi2);
}

static inline float4 fitPlaneBW(estimate* e, float4 meas, float2 mvar, float scatter, float dz){
  float4 var = mvar.s0101;
  predict(e,dz);
  addScatter(e, scatter);
  float4 chi2 = getChi2Inc(e, meas, var);
  update(e,meas, var);
  return(chi2);
}

__kernel void fitPlanes(__global float4* meas0,
			__global float4* meas1,
			__global float4* meas2,
			__global float4* meas3,
			__global float4* meas4,
			__global float4* meas5,
			__global float4* meas6,
			__global float4* meas7,
			__global float4* meas8,
			__global float4* chi2f_2,
			__global float4* chi2f_3,
			__global float4* chi2f_4,
			__global float4* chi2f_5,
			__global float4* chi2f_6,
			__global float4* chi2f_7,
			__global float4* chi2f_8,
			__global float4* chi2b_0,
			__global float4* chi2b_1,
			__global float4* chi2b_2,
			__global float4* chi2b_3,
			__global float4* chi2b_4,
			__global float4* chi2b_5,
			__global float4* chi2b_6,
			__constant float2* g_measvar,
			__constant float* g_invScattervar,
			__constant float* g_dz,
			__local float2* measvar,
			__local float* invScattervar,
			__local float* dz){
  int gid = get_global_id(0);
  int lid = get_local_id(0);

  estimate e = {(float4) {0.0f, 0.0f, 0.0f, 0.0f},
		(float4) {0.0f, 0.0f, 0.0f, 0.0f},
		(float4) {0.0f, 0.0f, 0.0f, 0.0f},
		(float4) {0.0f, 0.0f, 0.0f, 0.0f},
		(float4) {0.0f, 0.0f, 0.0f, 0.0f}};

  if(lid < 9){
    measvar[lid] = g_measvar[lid];
    invScattervar[lid] = g_invScattervar[lid];
    dz[lid] = g_dz[lid];
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  //Plane0
  float4 meas = meas0[gid];
  float2 mvar = measvar[0];
  float iScatter = invScattervar[0];
  update(&e,meas, mvar.s0101);
  addScatter(&e, iScatter);

  //Plane1
  meas = meas1[gid];
  mvar = measvar[1];
  iScatter = invScattervar[1];
  float mdz = dz[1];
  predict(&e,mdz);
  update(&e,meas, mvar.s0101);
  addScatter(&e, iScatter);
  //Plane2
  meas = meas2[gid];
  mvar = measvar[2];
  iScatter = invScattervar[2];
  mdz = dz[2];
  chi2f_2[gid] = fitPlaneFW(&e, meas, mvar, iScatter, mdz);
  //Plane3
  meas = meas3[gid];
  mvar = measvar[3];
  iScatter = invScattervar[3];
  mdz = dz[3];
  chi2f_3[gid] = fitPlaneFW(&e, meas, mvar, iScatter, mdz);
  //Plane4
  meas = meas4[gid];
  mvar = measvar[4];
  iScatter = invScattervar[4];
  mdz = dz[4];
  chi2f_4[gid] = fitPlaneFW(&e, meas, mvar, iScatter, mdz);
  //Plane5
  meas = meas5[gid];
  mvar = measvar[5];
  iScatter = invScattervar[5];
  mdz = dz[5];
  chi2f_5[gid] = fitPlaneFW(&e, meas, mvar, iScatter, mdz);
  //Plane6
  meas = meas6[gid];
  mvar = measvar[6];
  iScatter = invScattervar[6];
  mdz = dz[6];
  chi2f_6[gid] = fitPlaneFW(&e, meas, mvar, iScatter, mdz);
  //Plane7
  meas = meas7[gid];
  mvar = measvar[7];
  iScatter = invScattervar[7];
  mdz = dz[7];
  chi2f_7[gid] = fitPlaneFW(&e, meas, mvar, iScatter, mdz);
  //Plane8
  meas = meas8[gid];
  mvar = measvar[8];
  iScatter = invScattervar[8];
  mdz = dz[8];
  chi2f_8[gid] = fitPlaneFW(&e, meas, mvar, iScatter, mdz);
  //BW
  e.x    = (float4) {0.0f, 0.0f, 0.0f, 0.0f};
  e.dx   = (float4) {0.0f, 0.0f, 0.0f, 0.0f};
  e.xx   = (float4) {0.0f, 0.0f, 0.0f, 0.0f};
  e.xdx  = (float4) {0.0f, 0.0f, 0.0f, 0.0f};
  e.dxdx = (float4) {0.0f, 0.0f, 0.0f, 0.0f};
  
  //Plane8
  //meas = meas8[gid];
  //mvar = measvar[8];
  //iScatter = scattervar[8];
  update(&e, meas, mvar.s0101);
  addScatter(&e, iScatter);
  //Plane7
  meas = meas7[gid];
  mvar = measvar[7];
  iScatter = invScattervar[7];
  mdz = -dz[8];
  predict(&e,mdz);
  addScatter(&e, iScatter);
  update(&e,meas, mvar.s0101);
  //Plane6
  meas = meas6[gid];
  mvar = measvar[6];
  iScatter = invScattervar[6];
  mdz = -dz[7];
  chi2b_6[gid] = fitPlaneBW(&e, meas, mvar, iScatter, mdz);
  //Plane5
  meas = meas5[gid];
  mvar = measvar[5];
  iScatter = invScattervar[5];
  mdz = -dz[6];
  chi2b_5[gid] = fitPlaneBW(&e, meas, mvar, iScatter, mdz);
  //Plane4
  meas = meas4[gid];
  mvar = measvar[4];
  iScatter = invScattervar[4];
  mdz = -dz[5];
  chi2b_4[gid] = fitPlaneBW(&e, meas, mvar, iScatter, mdz);
  //Plane3
  meas = meas3[gid];
  mvar = measvar[3];
  iScatter = invScattervar[3];
  mdz = -dz[4];
  chi2b_3[gid] = fitPlaneBW(&e, meas, mvar, iScatter, mdz);
  //Plane2
  meas = meas2[gid];
  mvar = measvar[2];
  iScatter = invScattervar[2];
  mdz = -dz[3];
  chi2b_2[gid] = fitPlaneBW(&e, meas, mvar, iScatter, mdz);
  //Plane1
  meas = meas1[gid];
  mvar = measvar[1];
  iScatter = invScattervar[1];
  mdz = -dz[2];
  chi2b_1[gid] = fitPlaneBW(&e, meas, mvar, iScatter, mdz);
  //Plane0
  meas = meas0[gid];
  mvar = measvar[0];
  iScatter = invScattervar[0];
  mdz = -dz[1];
  chi2b_0[gid] = fitPlaneBW(&e, meas, mvar, iScatter, mdz);
}

//Reduces *buffer to an array of the same length as the number of workgroups
//Float2 is used, elements withe even indexes are chi2 increments in x, the odd are in y.
//The code is taken from an AMD example for reduce to min.
__kernel void reduce(__global float2* buffer,
		     __local float2* scratch,
		     __const int length,
		     __global float2* result) {
  int global_index = get_global_id(0);
  float2 accumulator = {0.0f, 0.0f};
  // Loop sequentially over chunks of input vector
  while (global_index < length) {
    float2 element = buffer[global_index];
    accumulator += element;
    global_index += get_global_size(0);
  }
  
  // Perform parallel reduction
  int local_index = get_local_id(0);
  scratch[local_index] = accumulator;
  barrier(CLK_LOCAL_MEM_FENCE);
  for(int offset = get_local_size(0) / 2;
      offset > 0;
      offset = offset / 2) {
    if (local_index < offset) {
      float2 other = scratch[local_index + offset];
      float2 mine = scratch[local_index];
      scratch[local_index] = mine + other;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (local_index == 0) {
    result[get_group_id(0)] = scratch[0];
  }
}
