/*a*****************************************************
a*         2D Quasi Acoustic VTI Medium  FD & RTM
b*  P + sv wave and get rid of sv
c*  GPU(CUDA) ,poynting adcigs, read shot
d*
e*******************************************************
f*
g* Ps:  the Quasi Acoustic VTI function:
h*
i*          du/dt=1/rho*dp/dx ,
j*          dw/dt=1/rho*dq/dz ,
k*          dp/dt=rho*vpx^2*du/dx+rho*vp*vpn*dw/dz ,
l*          dq/dt=rho*vp*vpn*du/dx+rho*vp^2*dw/dz ,
m*                     vpx^2=vp^2*(1+2*epsilon);
n*                     vpn^2=vp^2*(1+2*delta);
o*
p*******************************************************
q*                           initial: 2017.02 Rong Tao
r*                            adcigs: 2017.04 Rong Tao
s*                            modify: 2018.02 Rong Tao
u*                             print: 2018.05 Rong Tao
v*
w*
x*
y*******************************************************
z*/

#include <stdio.h>
#include <malloc.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <cuda_runtime.h>

#define ANSI_CYAN    "\x1b[36m"
#define ANSI_GREEN   "\x1b[32m"
#define ANSI_RED     "\x1b[31m"
#define ANSI_YELLOW  "\x1b[33m"
#define ANSI_RESET   "\x1b[0m"

/*print error message*/
void print_error(const char *message)
{
    if(message)
        printf(ANSI_RED "     [Error]  " ANSI_RESET "%s",message);
}
/*print warning message*/
void print_warning(const char *message)
{
    if(message )
        printf(ANSI_YELLOW "  [Warning]  "ANSI_RESET "%s",message);
}
/*printf info message*/
void print_info(const char *message)
{
    if(message )
        printf(ANSI_CYAN "     [Info]  "ANSI_RESET "%s", message);
}
/*print success message*/
void print_success(const char* message)
{
    if(message )
        printf(ANSI_GREEN "  [Success]  "ANSI_RESET "%s", message);
}
#ifdef pi
#pragma message("Already define pi !!!")
#else
#define pi 3.141592653
#endif

#define mm 4

#define Nbar 25

#define CHECK_gpu(call)  {                                          \
    const cudaError_t error = call;                             \
    char msg[1024];            \
    if (error != cudaSuccess)  {                                \
        sprintf(msg, "Error in: %s:%d: ", __FILE__, __LINE__);  \
        print_error(msg);                                       \
        sprintf(msg, "code: %d, reason: %s\n", error,        \
                cudaGetErrorString(error));                     \
        print_error(msg);                                       \
        exit(1);                                                \
    }                                            \
}

const char *note[] = {
"\n\033[1;31;44m       2D Quasi Acoustic VTI Medium  FD & RTM (CUDA, ADCIGs)     \033[m",
"\033[1;31;44m                               Author: Rong Tao @UPC             \033[m",
"                                                                               ",
" \033[1;31mQuasi Acoustic Function as follows:\033[m           ",
"    du/dt = dp/dx                                                       ",
"    dw/dt = dq/dz                                                       ",
"    dp/dt =  vpx^2 * du/dx + vp*vpn * dw/dz                             ",
"    dq/dt = vp*vpn * du/dx +  vp^2  * dw/dz                             ",
"    vpx^2 = vp^2 * (1+2*epsilon)                                        ",
"    vpn^2 = vp^2 * (1+2*delta)                                          ",
"                                                                               ",
" \033[1;31mRequired Parameters:\033[m                               ",
"    kind         =1 Finite difference forward modeling[FD]                       ",
"                 =2 Reverse Time Migration[RTM]                              ",
"    For example:                                                          ",
"    ./a.out  kind=1     Finite difference forward modeling[FD]                   ",
"    ./a.out  kind=2     Reverse Time Migration[RTM]                              ",
"                         no space in 'kind=1'                            ",
"                                                                               ",
" \033[1;31mInner Parameters:\033[m                                    ",
"    nx, dx       Horizontal Space sampling point and interval             ",
"    nz, dz       Vertical Space sampling point and interval               ",
"    nt, dt       Time sampling point and interval                         ",
"    favg         Wavelet frequency                                        ",
"    pfac         Wavelet Gain                                             ",
"    ns           The number of shots                                      ",
"    fs           First shot position[grid]                                ",
"    ds           Shots interval[grid]                                     ",
"    zs           Shots vertical position[grid]                            ",
"    nangle       The number of ADCIGs's angle                             ",
"    dangle       The interval of ADCIGs's angle                           ",
"    dAdcigs      Output file, the interval cdp(nx)                        ",
"    npml         PML Border width[grid]                                   ",
"                                                                               ",
" \033[1;31mOptional Parameters:\033[m                                   ",
"    wtype        kind of wavelet    =1 ricker wavelet                     ",
"                                    =2 derivative of gaussian             ",
"                                    =3 derivative of gaussian             ",
"    readShot     =true,             boolean, read obs shot                ",
"                 =false,            boolean, use accurate shot data       ",
"    writeSnap    =true,false        output snap into file or not          ",
"    ",
" \033[1;31mCOMMENTS\033[m                                                ",
"    Copyright (C) 2016-2018                                               ",
"            \033[1;32mChina University of Petroleum (East China), QingDao, China\033[m  ",
"    Authors: \033[1;32mRong Tao \033[m        ",
"    Location: \033[1;32mYi Fu Building @UPC\033[m                                        ",
"    Team: \033[1;32mLEON-VEL\033[m                                                       ",
"         ",
NULL
};


__device__ float d0;

__global__ void get_d0(float dx,
                       float dz,
                       int nnx,
                       int nnz,
                       int npml,
                       float *vp)
/* this (d0) function for pml bndr */
{
    d0 = 10.0*vp[nnx*nnz/2]*log(100000.0)/(2.0*npml*((dx+dz)/2.0));
}
/*#define mm 4*/
__constant__ float c[mm]={1.196289,-0.0797526,0.009570313,-0.0006975447};

void mBar(float fBar)
/* show progress bar */
{

    int i,j,k,m;
    //for ( i=0;i< Nbar+6; i++ )
    //    printf("\b");
    k = Nbar*fBar;
    m = fBar*100;
    printf("[");
    for ( i=0;i<k;i++ )
        printf("=");
    for ( j=0;j<Nbar-k;j++ )
        printf(" ");
    printf("]%3d%%",m);
}

void check_gpu_error (const char *msg)
/* check GPU errors */
{
    cudaError_t err = cudaGetLastError ();
    char message[1024];
    if (cudaSuccess != err) {
        sprintf(message,"Cuda error: %s: %s\n", msg, cudaGetErrorString(err));
        print_error(message);
        exit(0);
    }
    print_success("Check gpu successfully.\n");
}

void laplace_filter(int adj,
                    int nz,
                    int nx,
                    float *in,
                    float *out)
/**
 * linear operator
 *
 * Copyright@ Madagascar Mlaplac2
 */
{
    int iz,ix,j;
    for (j=0;j<nx*nz;j++)
        out[j]=0.0;

    for (ix=0; ix < nx; ix++) {
        for (iz=0; iz < nz; iz++) {
            j = iz+ix*nz;
            if (iz > 0) {
                if (adj) {
                    out[j-1] -= in[j];
                    out[j]   += in[j];
                } else {
                    out[j] += in[j] - in[j-1];
                }
            }
            if (iz < nz-1) {
                if (adj) {
                    out[j+1] -= in[j];
                    out[j]   += in[j];
                } else {
                    out[j] += in[j] - in[j+1];
                }
            }
            if (ix > 0) {
                if (adj) {
                    out[j-nz] -= in[j];
                    out[j]    += in[j];
                } else {
                    out[j] += in[j] - in[j-nz];
                }
            }
            if (ix < nx-1) {
                if (adj) {
                    out[j+nz] -= in[j];
                    out[j]    += in[j];
                } else {
                    out[j] += in[j] - in[j+nz];
                }
            }
        }
    }
}

__global__ void add_source( float pfac,
                            float xsn,
                            float zsn,
                            int nx,
                            int nz,
                            int nnx,
                            int nnz,
                            float dt,
                            float t,
                            float favg,
                            int wtype,
                            int npml,
                            int is,
                            int ds,
                            float *P,
                            float *Q)
/* generate ricker wavelet with time deley */
{
    int ixs,izs;
    float x_,xx_,tdelay,ts,source=0.0,fs;

    tdelay = 1.0/favg;
    ts = t-tdelay;
    fs = xsn+(is-1)*ds;

    if(wtype==1)//ricker wavelet
    {
        x_ = favg*ts;
        xx_ = x_*x_;
        source=(1-2*pi*pi*(xx_))*exp(-(pi*pi*xx_));

    }else if(wtype==2){//derivative of gaussian

        x_ = (-4)*favg*favg*pi*pi/log(0.1);
        source = (-2)*pi*pi*ts*exp(-x_*ts*ts);

    }else if(wtype==3){//derivative of gaussian

        x_ = (-1)*favg*favg*pi*pi/log(0.1);
        source = exp(-x_*ts*ts);
    }

    if(t <= 2*tdelay)
    {
        ixs = (int)( fs + 0.5) + npml - 1;
        izs = (int)(zsn + 0.5) + npml - 1;

        P[ixs*nnz+izs] += pfac * source;
        Q[ixs*nnz+izs] += pfac * source;
    }
}

__global__ void update_vel(int nx,
                           int nz,
                           int nnx,
                           int nnz,
                           int npml,
                           float dt,
                           float dx,
                           float dz,
                           float *u0,
                           float *w0,
                           float *u1,
                           float *w1,
                           float *P,
                           float *Q,
                           float *coffx1,
                           float *coffx2,
                           float *coffz1,
                           float *coffz2)
{
    int id=threadIdx.x+blockDim.x*blockIdx.x;

    int ix,iz,im;
    float dtx,dtz,xx,zz;

    ix = id/nnz;
    iz = id%nnz;

    dtx = dt/dx;
    dtz = dt/dz;

    if(id >= mm && id < nnx*nnz - mm) {

        if(ix >= mm && ix<(nnx-mm) && iz >= mm && iz<(nnz-mm)) {

            xx = 0.0;
            zz = 0.0;
            for(im = 0;im<mm;im++) {

                xx += c[im] * (P[id+(im+1)*nnz]  -  P[id-im*nnz]);
                zz += c[im] * (Q[id+im+1]        -  Q[id-im]);
            }
            u1[id] = coffx2[ix]*u0[id] - coffx1[ix]*dtx*xx;
            w1[id] = coffz2[iz]*w0[id] - coffz1[iz]*dtz*zz;
        }
    }
}

__global__ void update_stress(int nx,
                              int nz,
                              int nnx,
                              int nnz,
                              float dt,
                              float dx,
                              float dz,
                              float *u1,
                              float *w1,
                              float *P,
                              float *Q,
                              float *vp,
                              int npml,
                              float *px1,
                              float *px0,
                              float *pz1,
                              float *pz0,
                              float *qx1,
                              float *qx0,
                              float *qz1,
                              float *qz0,
                              float *acoffx1,
                              float *acoffx2,
                              float *acoffz1,
                              float *acoffz2,
                              float *delta,
                              float *epsilon,
                              int fs,
                              int ds,
                              int zs,
                              int is,
                              bool SV)
{
    int id=threadIdx.x+blockDim.x*blockIdx.x;

    int im, ix, iz, rx, rz;
    float dtx, dtz, xx, zz, ee, dd;

    /* iso circle */
    int R=18,r=7;

    ix = id / nnz;
    iz = id % nnz;

    dtx = dt / dx;
    dtz = dt / dz;

    if(id >= mm && id<nnx*nnz-mm) {

        /* iso circle begin */
        rx = ix-(fs+(is-1)*ds+npml);
        rz = iz-(zs+npml);

        if(SV){

            if((rx*rx+rz*rz) <= R*R){
                if((rx*rx+rz*rz) <= r*r){

                    ee = 0.0;
                    dd = 0.0;

                }else{

                    ee = 0.5*(1-cos(pi*((sqrtf(rx*rx+rz*rz)-r)*4.0/(R*3.0-1))))*epsilon[id];
                    dd = 0.5*(1-cos(pi*((sqrtf(rx*rx+rz*rz)-r)*4.0/(R*3.0-1))))*delta[id];

                }//else

            }else{

                    ee = epsilon[id];
                    dd = delta[id];
            }

        }else{

            ee = epsilon[id];
            dd = delta[id];

        }
        /* iso circle end */

        if(ix>=mm && ix<(nnx-mm) && iz>=mm && iz<(nnz-mm)) {

            xx=0.0;
            zz=0.0;

            for(im=0; im<mm; im++) {

                xx += c[im]*(u1[id+im*nnz] - u1[id-(im+1)*nnz]);
                zz += c[im]*(w1[id+im]     - w1[id-im-1]);
            }
            px1[id] = acoffx2[ix]*px0[id] - acoffx1[ix]*vp[id]*vp[id]*(1+2*ee)*dtx*xx;
            pz1[id] = acoffz2[iz]*pz0[id] - acoffz1[iz]*vp[id]*vp[id]*sqrtf(1+2*dd)*dtz*zz;
            qx1[id] = acoffx2[ix]*qx0[id] - acoffx1[ix]*vp[id]*vp[id]*sqrtf(1+2*dd)*dtx*xx;
            qz1[id] = acoffz2[iz]*qz0[id] - acoffz1[iz]*vp[id]*vp[id]*dtz*zz;

            P[id] = px1[id] + pz1[id];
            Q[id] = qx1[id] + qz1[id];
        }
    }
}

/**
 * Expand the border
 */
void pad_vv(int nx,
            int nz,
            int nnx,
            int nnz,
            int npml,
            float *ee)
{
    int ix,iz,id;

    for(id=0; id<nnx*nnz; id++) {

        ix = id/nnz;
        iz = id%nnz;

        /* left */
        if(ix<npml){

            ee[id] = ee[npml*nnz+iz];

        /* right */
        }else if(ix>=nnx-npml){

            ee[id] = ee[(nnx-npml-1)*nnz+iz];
        }
    }
    for(id=0; id<nnx*nnz; id++) {

        ix = id/nnz;
        iz = id%nnz;

        /* up */
        if(iz < npml){

            ee[id] = ee[ix*nnz+npml];

        /* bottom */
        }else if(iz >= nnz-npml){

            ee[id] = ee[ix*nnz+nnz-npml-1];
        }
    }
}

/**
 * Calculate the PML coefficient
 */
__global__ void initial_coffe(float dt,
                              int nn,
                              float *coff1,
                              float *coff2,
                              float *acoff1,
                              float *acoff2,
                              int npml)
{
    int id=threadIdx.x+blockDim.x*blockIdx.x;

    if(id < nn+2*npml) {

        /* The front of the inner */
        if(id<npml) {

            coff1[id] = 1.0/(1.0+(dt*d0*pow((npml-0.5-id)/npml,2.0))/2.0);
            coff2[id] = coff1[id]*(1.0-(dt*d0*pow((npml-0.5-id)/npml,2.0))/2.0);

            acoff1[id] = 1.0/(1.0+(dt*d0*pow(((npml-id)*1.0)/npml,2.0))/2.0);
            acoff2[id] = acoff1[id]*(1.0-(dt*d0*pow(((npml-id)*1.0)/npml,2.0))/2.0);

        /* media inner */
        }else if(id>=npml&&id<npml+nn){

            coff1[id] = 1.0;
            coff2[id] = 1.0;

            acoff1[id] = 1.0;
            acoff2[id] = 1.0;

        /* The tail of the inner */
        }else{

            coff1[id] = 1.0/(1.0+(dt*d0*pow((0.5+id-nn-npml)/npml,2.0))/2.0);
            coff2[id] = coff1[id]*(1.0-(dt*d0*pow((0.5+id-nn-npml)/npml,2.0))/2.0);

            acoff1[id] = 1.0/(1.0+(dt*d0*pow(((id-nn-npml)*1.0)/npml,2.0))/2.0);
            acoff2[id] = acoff1[id]*(1.0-(dt*d0*pow(((id-nn-npml)*1.0)/npml,2.0))/2.0);
        }
    }
}

/**
 * Record or load Receiver wavefield
 *      (nx) >> (nx,nt)
 *           or
 *   (nx,nt) >> (nx)
 */
__global__ void shot_record(int nnx,
                            int nnz,
                            int nx,
                            int nz,
                            int npml,
                            int it,
                            int nt,
                            float *P,
                            float *shot,
                            bool record)
{
    int id=threadIdx.x+blockDim.x*blockIdx.x;

    if(id<nx) {

        /* record the wavefield */
        if(record){

            shot[it+nt*id] = P[npml+nnz*(id+npml)];

        /* load the receiver wavefield */
        }else{

            P[npml+nnz*(id+npml)] = shot[it+nt*id];
        }
    }
}

/**
 * Record or backword the boundary wave field
 *
 */
__global__ void wavefield_bndr(int nnx,
                               int nnz,
                               int nx,
                               int nz,
                               int npml,
                               int it,
                               int nt,
                               float *P,
                               float *Q,
                               float *P_bndr,
                               float *Q_bndr,
                               bool record)
{
    int id=threadIdx.x+blockDim.x*blockIdx.x;

    if(id<2*nx+2*nz) {

        /* save boundary */
        if(record) {

            /* up */
            if(id<nx){

                P_bndr[it*(2*nx+2*nz)+id] = P[npml-1+nnz*(id+npml)];
                Q_bndr[it*(2*nx+2*nz)+id] = Q[npml-1+nnz*(id+npml)];

            /* bottom */
            }else if(id>=nx&&id<(2*nx)){

                P_bndr[it*(2*nx+2*nz)+id] = P[npml+nz+1+nnz*(id-nx+npml)];
                Q_bndr[it*(2*nx+2*nz)+id] = Q[npml+nz+1+nnz*(id-nx+npml)];

            /* left */
            }else if(id>=(2*nx)&&id<(2*nx+nz)){

                P_bndr[it*(2*nx+2*nz)+id] = P[id-2*nx+npml+nnz*(npml-1)];
                Q_bndr[it*(2*nx+2*nz)+id] = Q[id-2*nx+npml+nnz*(npml-1)];

            /* right */
            }else if(id>=(2*nx+nz)){

                P_bndr[it*(2*nx+2*nz)+id] = P[id-2*nx-nz+npml+nnz*(npml+nx+1)];
                Q_bndr[it*(2*nx+2*nz)+id] = Q[id-2*nx-nz+npml+nnz*(npml+nx+1)];

            }

        /* backward porpagation boundary */
        }else{

            /* up */
            if(id<nx){

                P[npml-1+nnz*(id+npml)] = P_bndr[it*(2*nx+2*nz)+id];
                Q[npml-1+nnz*(id+npml)] = Q_bndr[it*(2*nx+2*nz)+id];

            /* bottom */
            }else if(id>=nx&&id<(2*nx)){

                P[npml+nz+1+nnz*(id-nx+npml)] = P_bndr[it*(2*nx+2*nz)+id];
                Q[npml+nz+1+nnz*(id-nx+npml)] = Q_bndr[it*(2*nx+2*nz)+id];

            /* left */
            }else if(id>=(2*nx)&&id<(2*nx+nz)){

                P[id-2*nx+npml+nnz*(npml-1)] = P_bndr[it*(2*nx+2*nz)+id];
                Q[id-2*nx+npml+nnz*(npml-1)] = Q_bndr[it*(2*nx+2*nz)+id];

            /* right */
            }else if(id>=(2*nx+nz)){

                P[id-2*nx-nz+npml+nnz*(npml+nx+1)] = P_bndr[it*(2*nx+2*nz)+id];
                Q[id-2*nx-nz+npml+nnz*(npml+nx+1)] = Q_bndr[it*(2*nx+2*nz)+id];

            }
        }
    }
}

/**
 *mute direct waves
 */
__global__ void mute_directwave(int nx,
                                int nt,
                                float dt,
                                float favg,
                                float dx,
                                float dz,
                                int fs,
                                int ds,
                                int zs,
                                int is,
                                float *vp,
                                float *epsilon,
                                float *shot,
                                int tt)
{
    int it = threadIdx.x+blockDim.x*blockIdx.x;

    int mu_t, mu_nt;
    float mu_x, mu_z, mu_t0;

    int ix, id;

    for(ix = 0; ix < nx; ix ++){

        id = ix*nt + it;

        mu_x = dx*abs(ix-fs-(is-1)*ds);
        mu_z = dz*zs;
        mu_t0 = sqrtf(pow(mu_x,2)+pow(mu_z,2))/(vp[1]*sqrtf(1+2*epsilon[1]));
        mu_t = (int)(2.0/(dt*favg));
        mu_nt = (int)(mu_t0/dt)+mu_t+tt;

        if((it > (int)(mu_t0/dt)-tt) && (it<mu_nt))
            shot[id] = 0.0;
    }
}

/**
 * illumination matrix
 */
__global__ void cal_illumination(int nnx,
                                 int nnz,
                                 int nz,
                                 int npml,
                                 float *illumination,
                                 float *P,
                                 float *Q)
{
    int id = threadIdx.x+blockDim.x*blockIdx.x;
    int ix = id/nz;
    int iz = id%nz;

    if(id < nnx*nnz) {

        illumination[id] += P[iz+npml+nnz*(ix+npml)] * P[iz+npml+nnz*(ix+npml)]
                           +Q[iz+npml+nnz*(ix+npml)] * Q[iz+npml+nnz*(ix+npml)];

        if(illumination[id] <= 0.0 )
            illumination[id] = 1.0;
    }
}

/**
 * RTM migration
 */
__global__ void cal_migration(int nnx,
                              int nnz,
                              int nz,
                              int npml,
                              float *migration,
                              float *s,
                              float *g)
{
    int id = threadIdx.x+blockDim.x*blockIdx.x;
    int ix = id/nz;
    int iz = id%nz;

    if(id<nnx*nnz) {

        migration[id] += s[iz+npml+nnz*(ix+npml)] * g[iz+npml+nnz*(ix+npml)];
    }
}

/**
 *  illuminate
 */
__global__ void migration_illum(int nx,
                                int nz,
                                int npml,
                                float *migration,
                                float *illumination)
{
    int id=threadIdx.x+blockDim.x*blockIdx.x;

    if(id<nx*nz) {

        migration[id] /= illumination[id];
    }
}

/**
 *  poynting vector extraction ADCIGs
 *    "Copyright(C) Madagascar:user/pyang/Mrtmadcig.c"
 */
__global__ void Poynting_Adcigs(int nnz,
                                int nx,
                                int nz,
                                int npml,
                                int nangle,
                                int dangle,
                                float *adcigs,
                                float *s_P,
                                float *s_Q,
                                float *s_u,
                                float *s_w,
                                float *g_P,
                                float *g_Q,
                                float *g_u,
                                float *g_w)
{
    int id = threadIdx.x+blockDim.x*blockIdx.x;
    int ix = id/nz;
    int iz = id%nz;

    int ia = 0;

    float Ssx = -s_P[iz+npml+nnz*(ix+npml)]*s_u[iz+npml+nnz*(ix+npml)];
    float Ssz = -s_Q[iz+npml+nnz*(ix+npml)]*s_w[iz+npml+nnz*(ix+npml)];
    float Sgx =  g_P[iz+npml+nnz*(ix+npml)]*g_u[iz+npml+nnz*(ix+npml)];
    float Sgz =  g_Q[iz+npml+nnz*(ix+npml)]*g_w[iz+npml+nnz*(ix+npml)];

    float b1 =  Ssx*Ssx + Ssz*Ssz;
    float b2 =  Sgx*Sgx + Sgz*Sgz;
    float  a = (Ssx*Sgx + Ssz*Sgz)/(sqrtf(b1*b2)*(1 - 0.1));

    if(id<nx*nz) {

        if(a>=-1&&a<=1) {

          a = 0.5*acosf(a)*180.0/pi;
         ia = (int)(a/(dangle*1.0));

            if(ia<nangle) {
                adcigs[iz+nz*ia+nz*nangle*(id/nz)]
                    += s_P[iz+npml+nnz*(ix+npml)]*g_P[iz+npml+nnz*(ix+npml)]
                      *cosf(ia*pi/180.0)*cosf(ia*pi/180.0)*cosf(ia*pi/180.0);
            }
        }
    }
}

/**
 *  illuminate the adcigs
 */
__global__ void adcigs_illum(int nx,
                             int nz,
                             int nangle,
                             int dangle,
                             float *adcigs,
                             float *illumination)
{
    int id = threadIdx.x+blockDim.x*blockIdx.x;
    int ix = id/(nz*nangle);
    int iz = id%nz;

    if(id<nx*nz*nangle) {

        adcigs[id] /= illumination[iz+nz*ix];
    }
}

/**
 * Stack adcigs to migration
 * Can suppress low-frequency random noise
 */
void stk_adcigs(int nx,
                int nz,
                int nangle,
                float *adcigs,
                float *migration)
{
    int ix,iz,ia,id,ido;
    float stk;
    float *temp;

    temp=(float*)malloc(nz*nx*sizeof(float));

    for (ix=0; ix<nx; ix++)  {
        for (iz=0; iz<nz; iz++)  {
            stk=0.0;
            for (ia=0; ia<nangle; ia++)  {
                id = ix*nangle*nz+ia*nz+iz;
                stk += adcigs[id];
            }
            ido = ix*nz+iz;
            temp[ido] = stk;
        }
    }
    laplace_filter(1,nz,nx,temp,migration);
    print_success("Done with stack ADCIGs into Migration.\n");
}

/**
 * Draw thin adcigs
 */
void adcigs_smiled(int nx,
                   int nz,
                   int nangle,
                   int dAdcigs,
                   float *adcigs)
{
    int ix,iz,ia,id,ido;
    float *temp;

    temp = (float*)malloc(nz*nx/dAdcigs*nangle*sizeof(float));

    for (ix=0; ix<nx; ix++)  {
        for (ia=0; ia<nangle; ia++)  {
            for (iz=0; iz<nz; iz++)  {

                id=ix*nangle*nz+ia*nz+iz;

                if(ix%dAdcigs==0) {

                    ido = ix/dAdcigs*nangle*nz+ia*nz+iz;
                    temp[ido] = adcigs[id];
                    adcigs[ido] = temp[ido];
                }
            }
        }
    }
    print_success("Done with smile ADCIGs.\n");
}

void readFile( char FNvelocity[],
               char FNepsilon[],
               char FNdelta[],
               int nx,
               int nz,
               int nnx,
               int nnz,
               float dx,
               float dz,
               float favg,
               float dt,
               float *v,
               float *e,
               float *d,
               int npml)
{
    int i,j,id;
    float vmax, vmin;
    float emax, emin;
    float dmax, dmin;
    float H_min, dt_max, dxz_max, C, tmp;
    char msg[1024];
    FILE *fp1,*fp2,*fp3;


    if((fp1=fopen(FNvelocity,"rb"))==NULL){

        sprintf(msg, "error open <%s>!\n",FNvelocity);
        print_error(msg);
        exit(0);
    }
    if((fp2=fopen(FNepsilon,"rb"))==NULL){

        sprintf(msg, "error open <%s>!\n",FNepsilon);
        print_error(msg);
        exit(0);
    }
    if((fp3=fopen(FNdelta,"rb"))==NULL){

        sprintf(msg, "error open <%s>!\n",FNdelta);
        print_error(msg);
        exit(0);
    }

    vmin = emin = dmin =  999999.9;
    vmax = emax = dmax = -999999.9;

    for(i=npml;i<nx+npml;i++) {
        for(j=npml;j<nz+npml;j++) {

            id=i*nnz+j;
                                  /* inch time 0.3 */
            fread(&v[id],4L,1,fp1);//v[id] *= 0.3;
            fread(&e[id],4L,1,fp2);
            fread(&d[id],4L,1,fp3);

            /* For Parameters Sensitivity Analysis */
            //if(true) // true: active
           // if(v[id]>3800)
            {

                //v[id] *= 0.3;
                //e[id] *= 0.85;
                //d[id] *= 0.85;
            }

            if(vmax<v[id]) vmax = v[id];
            if(vmin>v[id]) vmin = v[id];
            if(emax<e[id]) emax = e[id];
            if(emin>e[id]) emin = e[id];
            if(dmax<d[id]) dmax = d[id];
            if(dmin>d[id]) dmin = d[id];
        }
    }
    fclose(fp1);
    fclose(fp2);
    fclose(fp3);

    print_success("Load (v,e,d) successfully.\n");

    sprintf(msg,"   Velocity Range (%.1f - %.1f)[m/s]\n",vmin,vmax);
    print_info(msg);
    sprintf(msg,"    Epsilon Range (%.4f - %.4f)\n",emin,emax);
    print_info(msg);
    sprintf(msg,"      Delta Range (%.4f - %.4f)\n",dmin,dmax);
    print_info(msg);

    /* boundary */
    pad_vv(nx,nz,nnx,nnz,npml,e);
    pad_vv(nx,nz,nnx,nnz,npml,d);
    pad_vv(nx,nz,nnx,nnz,npml,v);

    H_min=dx<dz?dx:dz;
    dt_max = 0.5*H_min/vmin;
    dxz_max = vmax/favg*0.2;

    if ( dxz_max<dz || dxz_max<dx){
        print_warning("You need have to redefine DX and DZ ! \n");
        exit(0);
    }
    if ( dt_max<dt){
        print_warning("You need have to redefine DT ! \n");
        exit(0);
    }
    if ( favg >= vmin/( 5.0*(dx>dz?dx:dz) )
      || favg >= vmin/( 5.0*(dx>dz?dx:dz) ) ) {
        print_warning("Non-dispersion relation not satisfied! \n");
        exit(0);
    }
    /* following
     * Copyright@ Madagascar */
    else if ( mm == 2 )
        C = 0.857;
    else if ( mm == 3 )
        C = 0.8;
    else if ( mm == 4 )
        C = 0.777;
    else if ( mm == 5 )
        C = 0.759;

    tmp = dt*vmax*sqrtf( 1.0/(dx*dx)+1.0/(dz*dz) );
    if ( tmp >= C){

        sprintf(msg, "Stability condition not satisfied! tmp = %f, C = %f\n",tmp,C);
        print_warning(msg);
        exit(0);
    }
}

/**
 * FD
 *         Author: Rong Tao
 */
void FD( char FNvelocity[],
         char FNepsilon[],
         char FNdelta[],
         char FNCalShot[],
         char FNSnap[],
         char FNIllumination[],
         int wtype,
         int npml,
         int nx,
         int nz,
         float dx,
         float dz,
         int nt,
         float dt,
         int ns,
         int fs,
         int ds,
         int zs,
         float favg,
         float pfac,
         bool writeSnap)
{
    float *v, *e, *d;
    float *vp, *epsilon, *delta;

    float *s_u0, *s_u1, *s_px0, *s_qx0, *s_px1, *s_qx1;
    float *s_w0, *s_w1, *s_pz0, *s_qz0, *s_pz1, *s_qz1;

    float *s_P, *s_Q;

    float *coffx1,*coffx2,*coffz1,*coffz2;
    float *acoffx1,*acoffx2,*acoffz1,*acoffz2;

    float *shot_Dev, *shot_Hos;
    float *illumination;

    int nnx, nnz;
    int it, is;

    float t;

    /* FILE pointer */
    FILE *fpCalShot = fopen(FNCalShot,"wb");
    FILE *fpSnap;
    if(writeSnap) {

        fpSnap = fopen(FNSnap,"wb");
    }
    FILE *fpIllunmination = fopen(FNIllumination,"wb");

    /* whole media size */
    nnx = nx + 2*npml;
    nnz = nz + 2*npml;

    /* read the media file into memory */
    v = (float*)malloc(nnz*nnx*sizeof(float));
    e = (float*)malloc(nnz*nnx*sizeof(float));
    d = (float*)malloc(nnz*nnx*sizeof(float));
    readFile(FNvelocity,FNepsilon,FNdelta,
             nx,nz,nnx,nnz,dx,dz,favg,dt,v,e,d,npml);

    /* alloc host and device record memory */
    shot_Hos=(float*)malloc(nt*nx*sizeof(float));

    /* initialize device, default device=0; */
    cudaSetDevice(0);
    check_gpu_error("Failed to initialize device!");
    CHECK_gpu(cudaDeviceReset());

    /* malloc the device media memory */
    cudaMalloc(&vp, nnz*nnx*sizeof(float));
    cudaMalloc(&epsilon, nnz*nnx*sizeof(float));
    cudaMalloc(&delta, nnz*nnx*sizeof(float));

    /* copy the media parameters host memory to device */
    cudaMemcpy(vp, v, nnz*nnx*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(epsilon, e, nnz*nnx*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(delta, d, nnz*nnx*sizeof(float), cudaMemcpyHostToDevice);

    /* source wavefield device memory */
    cudaMalloc(&s_u0, nnz*nnx*sizeof(float)); cudaMalloc(&s_u1, nnz*nnx*sizeof(float));
    cudaMalloc(&s_w0, nnz*nnx*sizeof(float)); cudaMalloc(&s_w1, nnz*nnx*sizeof(float));
    cudaMalloc(&s_P, nnz*nnx*sizeof(float));  cudaMalloc(&s_Q, nnz*nnx*sizeof(float));
    cudaMalloc(&s_px0, nnz*nnx*sizeof(float));cudaMalloc(&s_px1, nnz*nnx*sizeof(float));
    cudaMalloc(&s_pz0, nnz*nnx*sizeof(float));cudaMalloc(&s_pz1, nnz*nnx*sizeof(float));
    cudaMalloc(&s_qx0, nnz*nnx*sizeof(float));cudaMalloc(&s_qx1, nnz*nnx*sizeof(float));
    cudaMalloc(&s_qz0, nnz*nnx*sizeof(float));cudaMalloc(&s_qz1, nnz*nnx*sizeof(float));

    /* boundary absorb coefficient device memory */
    cudaMalloc(&coffx1, nnx*sizeof(float));   cudaMalloc(&acoffx1, nnx*sizeof(float));
    cudaMalloc(&coffx2, nnx*sizeof(float));   cudaMalloc(&acoffx2, nnx*sizeof(float));
    cudaMalloc(&coffz1, nnz*sizeof(float));   cudaMalloc(&acoffz1, nnz*sizeof(float));
    cudaMalloc(&coffz2, nnz*sizeof(float));   cudaMalloc(&acoffz2, nnz*sizeof(float));

    cudaMalloc(&shot_Dev, nx*nt*sizeof(float));

    /* imaging device memory */
    cudaMalloc(&illumination, nz*nx*sizeof(float));

    /* check Nvidia GPU */
    check_gpu_error("Failed to allocate memory for variables!");

    /* calculate d0 and pml adsorb coffe */
    get_d0<<<1, 1>>>(dx, dz, nnx, nnz, npml, vp);
    initial_coffe<<<(nnx+511)/512, 512>>>(dt,nx,coffx1,coffx2,acoffx1,acoffx2,npml);
    initial_coffe<<<(nnz+511)/512, 512>>>(dt,nz,coffz1,coffz2,acoffz1,acoffz2,npml);

    /* set Imaging to zero */
    cudaMemset(illumination, 0, nz*nx*sizeof(float));

    clock_t time;

    char msg[1024];
    /* Starting IS loop */
    for(is=1; is<=ns; is++)	 {

        time = clock();
        sprintf(msg, "IS =%3d/%d ",is,ns);
        print_info(msg);
        mBar(1.0*is/(1.0*ns));

        cudaMemset(s_u0, 0, nnz*nnx*sizeof(float));  cudaMemset(s_u1, 0, nnz*nnx*sizeof(float));
        cudaMemset(s_w0, 0, nnz*nnx*sizeof(float));  cudaMemset(s_w1, 0, nnz*nnx*sizeof(float));
        cudaMemset(s_P, 0, nnz*nnx*sizeof(float));   cudaMemset(s_Q, 0, nnz*nnx*sizeof(float));
        cudaMemset(s_px0, 0, nnz*nnx*sizeof(float)); cudaMemset(s_px1, 0, nnz*nnx*sizeof(float));
        cudaMemset(s_pz0, 0, nnz*nnx*sizeof(float)); cudaMemset(s_pz1, 0, nnz*nnx*sizeof(float));
        cudaMemset(s_qx0, 0, nnz*nnx*sizeof(float)); cudaMemset(s_qx1, 0, nnz*nnx*sizeof(float));
        cudaMemset(s_qz0, 0, nnz*nnx*sizeof(float)); cudaMemset(s_qz1, 0, nnz*nnx*sizeof(float));

        cudaMemset(shot_Dev, 0, nt*nx*sizeof(float));

        /* forward */
        for(it=0,t=dt; it<nt; it++,t+=dt) {

            add_source<<<1,1>>>(pfac, fs,zs,nx,nz,nnx,nnz,dt,t,favg,wtype,npml,is,ds,s_P,s_Q);
            update_vel<<<(nnx*nnz+511)/512, 512>>>
                        (nx,nz,nnx,nnz,npml,dt,dx,dz,
                         s_u0,s_w0,s_u1,s_w1,s_P,s_Q,coffx1,coffx2,coffz1,coffz2);
            update_stress<<<(nnx*nnz+511)/512, 512>>>
                          (nx,nz,nnx,nnz,dt,dx,dz,s_u1,s_w1,s_P,s_Q,vp,npml,
                           s_px1,s_px0,s_pz1,s_pz0,s_qx1,s_qx0,s_qz1,s_qz0,
                           acoffx1,acoffx2,acoffz1,acoffz2,delta,epsilon,fs,ds,zs,is,false);
            s_u0 = s_u1;   s_w0 = s_w1;
            s_px0 = s_px1; s_pz0 = s_pz1;
            s_qx0 = s_qx1; s_qz0 = s_qz1;

            shot_record<<<(nx+511)/512, 512>>>
                        (nnx, nnz, nx, nz, npml, it, nt, s_P, shot_Dev, true);
            cal_illumination<<<(nx*nz+511)/512, 512>>>
                        (nnx, nnz, nz, npml, illumination, s_P, s_Q);

            if(writeSnap && (it%300==0)) {
                cudaMemcpy(e, s_P, nnz*nnx*sizeof(float), cudaMemcpyDeviceToHost);
                for(int i = npml; i<nnx-npml; i++)
                    for(int j = npml; j<nnz-npml; j++)
                        fwrite(&e[i*nnz+j], 4L, 1, fpSnap);
            }

        }//it

        mute_directwave<<<(nt+511)/512, 512>>>
                        (nx,nt,dt,favg,dx,dz,fs,ds,zs,is,vp,epsilon,shot_Dev,100);

        cudaMemcpy(shot_Hos, shot_Dev, nt*nx*sizeof(float), cudaMemcpyDeviceToHost);
        fwrite(shot_Hos,sizeof(float),nt*nx,fpCalShot);

        time = clock() - time;
        printf(", %f min\n", ((float)time)/60.0/CLOCKS_PER_SEC);

    }//is

    /* output multi-shot illumination */
    cudaMemcpy(e, illumination, nz*nx*sizeof(float), cudaMemcpyDeviceToHost);
    fwrite(e,sizeof(float),nx*nz,fpIllunmination);


    /* file close */
    if(writeSnap)
        fclose(fpSnap);
    fclose(fpCalShot);
    fclose(fpIllunmination);

    /* device memory free */
    cudaFree(coffx1);     cudaFree(acoffx1);
    cudaFree(coffx2);     cudaFree(acoffx2);
    cudaFree(coffz1);     cudaFree(acoffz1);
    cudaFree(coffz2);     cudaFree(acoffz2);

    cudaFree(s_u0);       cudaFree(s_u1);
    cudaFree(s_w0);       cudaFree(s_w1);
    cudaFree(s_P);        cudaFree(s_Q);
    cudaFree(s_px0);      cudaFree(s_px1);
    cudaFree(s_pz0);      cudaFree(s_pz1);
    cudaFree(s_qx0);      cudaFree(s_qx1);
    cudaFree(s_qz0);      cudaFree(s_qz1);

    cudaFree(shot_Dev);

    cudaFree(illumination);

    /* host memory free */
    free(v);
    free(e);
    free(d);
    free(shot_Hos);

    print_success("Finish Finite Difference\n");
}//FD

/**
 * RTM
 *         Author: Rong Tao
 */
void RTM(char FNvelocity[],
         char FNepsilon[],
         char FNdelta[],
         char FNObsShot[],
         char FNCalShot[],
         char FNSnap[],
         char FNMigration[],
         char FNIllumination[],
         char FNAdcigs[],
         char FNStkAdcigs[],
         char FNIntervalAdcigs[],
         int wtype,
         int npml,
         int nx,
         int nz,
         float dx,
         float dz,
         int nt,
         float dt,
         int ns,
         int fs,
         int ds,
         int zs,
         float favg,
         float pfac,
         int nangle,
         int dangle,
         int dAdcigs,
         bool readShot,
         bool writeSnap)
{
    float *v, *e, *d;
    float *vp, *epsilon, *delta;

    float *s_u0, *s_u1, *s_px0, *s_qx0, *s_px1, *s_qx1;
    float *s_w0, *s_w1, *s_pz0, *s_qz0, *s_pz1, *s_qz1;
    float *g_u0, *g_u1, *g_px0, *g_qx0, *g_px1, *g_qx1;
    float *g_w0, *g_w1, *g_pz0, *g_qz0, *g_pz1, *g_qz1;

    float *s_P, *s_Q, *g_P, *g_Q;

    float *coffx1,*coffx2,*coffz1,*coffz2;
    float *acoffx1,*acoffx2,*acoffz1,*acoffz2;

    float *shot_Dev, *shot_Hos, *P_bndr, *Q_bndr;
    float *migration, *illumination, *adcigs;
    float *Atemp;

    int nnx, nnz;
    int it, is;

    float t;

    /* FILE pointer */
    FILE *fpObsShot, *fpCalShot;
    if(readShot) {

        if((fpObsShot = fopen(FNObsShot,"rb"))==NULL){
            print_error(" ");
            printf("error open <%s>!\n",FNObsShot);
            exit(0);
        }
    }else{

        fpCalShot = fopen(FNCalShot,"wb");
    }
    FILE *fpSnap;
    if(writeSnap) {

        fpSnap = fopen(FNSnap,"wb");
    }
    FILE *fpMigration         = fopen(FNMigration,"wb");
    FILE *fpIllunmination     = fopen(FNIllumination,"wb");
    FILE *fpAdcigs            = fopen(FNAdcigs,"wb");
    FILE *fpStkAdcigs         = fopen(FNStkAdcigs,"wb");
    FILE *fpIntervalAdcigs    = fopen(FNIntervalAdcigs,"wb");

    /* whole media size */
    nnx = nx + 2*npml;
    nnz = nz + 2*npml;

    /* temp malloc for host adcigs */
    Atemp = (float*)malloc(nz*nx*nangle*sizeof(float));

    /* read the media file into memory */
    v = (float*)malloc(nnz*nnx*sizeof(float));
    e = (float*)malloc(nnz*nnx*sizeof(float));
    d = (float*)malloc(nnz*nnx*sizeof(float));
    readFile(FNvelocity,FNepsilon,FNdelta,
             nx,nz,nnx,nnz,dx,dz,favg,dt,v,e,d,npml);

    /* alloc host and device record memory */
    shot_Hos=(float*)malloc(nt*nx*sizeof(float));

    /* initialize device, default device=0; */
    cudaSetDevice(0);
    check_gpu_error("Failed to initialize device!");
    CHECK_gpu(cudaDeviceReset());

    /* malloc the device media memory */
    cudaMalloc(&vp, nnz*nnx*sizeof(float));
    cudaMalloc(&epsilon, nnz*nnx*sizeof(float));
    cudaMalloc(&delta, nnz*nnx*sizeof(float));

    /* copy the media parameters host memory to device */
    cudaMemcpy(vp, v, nnz*nnx*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(epsilon, e, nnz*nnx*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(delta, d, nnz*nnx*sizeof(float), cudaMemcpyHostToDevice);

    /* source wavefield device memory */      /* receiver wavefield device memory */
    cudaMalloc(&s_u0, nnz*nnx*sizeof(float));  cudaMalloc(&g_u0, nnz*nnx*sizeof(float));
    cudaMalloc(&s_u1, nnz*nnx*sizeof(float));  cudaMalloc(&g_u1, nnz*nnx*sizeof(float));
    cudaMalloc(&s_w0, nnz*nnx*sizeof(float));  cudaMalloc(&g_w0, nnz*nnx*sizeof(float));
    cudaMalloc(&s_w1, nnz*nnx*sizeof(float));  cudaMalloc(&g_w1, nnz*nnx*sizeof(float));

    cudaMalloc(&s_P, nnz*nnx*sizeof(float));   cudaMalloc(&g_P, nnz*nnx*sizeof(float));
    cudaMalloc(&s_Q, nnz*nnx*sizeof(float));   cudaMalloc(&g_Q, nnz*nnx*sizeof(float));

    cudaMalloc(&s_px0, nnz*nnx*sizeof(float)); cudaMalloc(&g_px0, nnz*nnx*sizeof(float));
    cudaMalloc(&s_px1, nnz*nnx*sizeof(float)); cudaMalloc(&g_px1, nnz*nnx*sizeof(float));
    cudaMalloc(&s_pz0, nnz*nnx*sizeof(float)); cudaMalloc(&g_pz0, nnz*nnx*sizeof(float));
    cudaMalloc(&s_pz1, nnz*nnx*sizeof(float)); cudaMalloc(&g_pz1, nnz*nnx*sizeof(float));
    cudaMalloc(&s_qx0, nnz*nnx*sizeof(float)); cudaMalloc(&g_qx0, nnz*nnx*sizeof(float));
    cudaMalloc(&s_qx1, nnz*nnx*sizeof(float)); cudaMalloc(&g_qx1, nnz*nnx*sizeof(float));
    cudaMalloc(&s_qz0, nnz*nnx*sizeof(float)); cudaMalloc(&g_qz0, nnz*nnx*sizeof(float));
    cudaMalloc(&s_qz1, nnz*nnx*sizeof(float)); cudaMalloc(&g_qz1, nnz*nnx*sizeof(float));

    /* boundary absorb coefficient device memory */
    cudaMalloc(&coffx1, nnx*sizeof(float));    cudaMalloc(&acoffx1, nnx*sizeof(float));
    cudaMalloc(&coffx2, nnx*sizeof(float));    cudaMalloc(&acoffx2, nnx*sizeof(float));
    cudaMalloc(&coffz1, nnz*sizeof(float));    cudaMalloc(&acoffz1, nnz*sizeof(float));
    cudaMalloc(&coffz2, nnz*sizeof(float));    cudaMalloc(&acoffz2, nnz*sizeof(float));

    /* boundary wavefield device memory */
    cudaMalloc(&P_bndr, nt*(2*nx+2*nz)*sizeof(float));
    cudaMalloc(&Q_bndr, nt*(2*nx+2*nz)*sizeof(float));

    cudaMalloc(&shot_Dev, nx*nt*sizeof(float));

    /* imaging device memory */
    cudaMalloc(&migration, nz*nx*sizeof(float));
    cudaMalloc(&illumination, nz*nx*sizeof(float));
    cudaMalloc(&adcigs, nz*nangle*nx*sizeof(float));

    /* check Nvidia GPU */
    check_gpu_error("Failed to allocate memory for variables!");

    /* calculate d0 and pml adsorb coffe */
    get_d0<<<1, 1>>>(dx, dz, nnx, nnz, npml, vp);
    initial_coffe<<<(nnx+511)/512, 512>>>(dt,nx,coffx1,coffx2,acoffx1,acoffx2,npml);
    initial_coffe<<<(nnz+511)/512, 512>>>(dt,nz,coffz1,coffz2,acoffz1,acoffz2,npml);

    /* set Imaging to zero */
    cudaMemset(migration, 0, nz*nx*sizeof(float));
    cudaMemset(illumination, 0, nz*nx*sizeof(float));
    cudaMemset(adcigs, 0, nz*nangle*nx*sizeof(float));

    clock_t time;

    char msg[1024];
    /* Starting IS loop */
    for(is=1; is<=ns; is++)	 {

        time = clock();
        sprintf(msg, "IS =%3d/%d ",is,ns);
        print_info(msg);
        mBar(1.0*is/(1.0*ns));

        cudaMemset(s_u0, 0, nnz*nnx*sizeof(float)); cudaMemset(g_u0, 0, nnz*nnx*sizeof(float));
        cudaMemset(s_u1, 0, nnz*nnx*sizeof(float)); cudaMemset(g_u1, 0, nnz*nnx*sizeof(float));
        cudaMemset(s_w0, 0, nnz*nnx*sizeof(float)); cudaMemset(g_w0, 0, nnz*nnx*sizeof(float));
        cudaMemset(s_w1, 0, nnz*nnx*sizeof(float)); cudaMemset(g_w1, 0, nnz*nnx*sizeof(float));

        cudaMemset(s_P, 0, nnz*nnx*sizeof(float));  cudaMemset(g_P, 0, nnz*nnx*sizeof(float));
        cudaMemset(s_Q, 0, nnz*nnx*sizeof(float));  cudaMemset(g_Q, 0, nnz*nnx*sizeof(float));

        cudaMemset(s_px0, 0, nnz*nnx*sizeof(float));cudaMemset(g_px0, 0, nnz*nnx*sizeof(float));
        cudaMemset(s_px1, 0, nnz*nnx*sizeof(float));cudaMemset(g_px1, 0, nnz*nnx*sizeof(float));
        cudaMemset(s_pz0, 0, nnz*nnx*sizeof(float));cudaMemset(g_pz0, 0, nnz*nnx*sizeof(float));
        cudaMemset(s_pz1, 0, nnz*nnx*sizeof(float));cudaMemset(g_pz1, 0, nnz*nnx*sizeof(float));
        cudaMemset(s_qx0, 0, nnz*nnx*sizeof(float));cudaMemset(g_qx0, 0, nnz*nnx*sizeof(float));
        cudaMemset(s_qx1, 0, nnz*nnx*sizeof(float));cudaMemset(g_qx1, 0, nnz*nnx*sizeof(float));
        cudaMemset(s_qz0, 0, nnz*nnx*sizeof(float));cudaMemset(g_qz0, 0, nnz*nnx*sizeof(float));
        cudaMemset(s_qz1, 0, nnz*nnx*sizeof(float));cudaMemset(g_qz1, 0, nnz*nnx*sizeof(float));

        cudaMemset(shot_Dev, 0, nt*nx*sizeof(float));
        cudaMemset(P_bndr, 0, nt*(2*nx+2*nz)*sizeof(float));
        cudaMemset(Q_bndr, 0, nt*(2*nx+2*nz)*sizeof(float));

        /* forward */
        for(it=0,t=dt; it<nt; it++,t+=dt) {

            add_source<<<1,1>>>(pfac, fs,zs,nx,nz,nnx,nnz,dt,t,favg,wtype,npml,is,ds,s_P,s_Q);
            update_vel<<<(nnx*nnz+511)/512, 512>>>
                        (nx,nz,nnx,nnz,npml,dt,dx,dz,
                         s_u0,s_w0,s_u1,s_w1,s_P,s_Q,coffx1,coffx2,coffz1,coffz2);
            update_stress<<<(nnx*nnz+511)/512, 512>>>
                          (nx,nz,nnx,nnz,dt,dx,dz,s_u1,s_w1,s_P,s_Q,vp,npml,
                           s_px1,s_px0,s_pz1,s_pz0,s_qx1,s_qx0,s_qz1,s_qz0,
                           acoffx1,acoffx2,acoffz1,acoffz2,delta,epsilon,fs,ds,zs,is,true);
            s_u0 = s_u1;   s_w0 = s_w1;
            s_px0 = s_px1; s_pz0 = s_pz1;
            s_qx0 = s_qx1; s_qz0 = s_qz1;

            shot_record<<<(nx+511)/512, 512>>>
                        (nnx, nnz, nx, nz, npml, it, nt, s_P, shot_Dev, true);
            wavefield_bndr<<<((2*nx+2*nz)+511)/512,512>>>
                        (nnx, nnz, nx, nz, npml, it, nt, s_P, s_Q, P_bndr, Q_bndr, true);
            cal_illumination<<<(nx*nz+511)/512, 512>>>
                        (nnx, nnz, nz, npml, illumination, s_P, s_Q);

            if(writeSnap && (it%300==0)) {
                cudaMemcpy(e, s_P, nnz*nnx*sizeof(float), cudaMemcpyDeviceToHost);
                for(int i = npml; i<nnx-npml; i++)
                    for(int j = npml; j<nnz-npml; j++)
                        fwrite(&e[i*nnz+j], 4L, 1, fpSnap);
            }

        }//it

        mute_directwave<<<(nt+511)/512, 512>>>
                        (nx,nt,dt,favg,dx,dz,fs,ds,zs,is,vp,epsilon,shot_Dev,100);

        if(readShot) {

            fread(shot_Hos,sizeof(float),nt*nx,fpObsShot);
            cudaMemcpy(shot_Dev, shot_Hos, nt*nx*sizeof(float), cudaMemcpyHostToDevice);

        } else {

            cudaMemcpy(shot_Hos, shot_Dev, nt*nx*sizeof(float), cudaMemcpyDeviceToHost);
            fwrite(shot_Hos,sizeof(float),nt*nx,fpCalShot);
        }

        /* backward */
        for(it=nt-1; it>=0; it--) {

            /* source wavefield */
            wavefield_bndr<<<((2*nx+2*nz)+511)/512,512>>>
                            (nnx, nnz, nx, nz, npml, it, nt, s_P, s_Q, P_bndr, Q_bndr, false);
            update_vel<<<(nnx*nnz+511)/512, 512>>>
                            (nx,nz,nnx,nnz,npml,dt,dx,dz,
                             s_u0,s_w0,s_u1,s_w1,s_P,s_Q,coffx1,coffx2,coffz1,coffz2);
            update_stress<<<(nnx*nnz+511)/512, 512>>>
                            (nx,nz,nnx,nnz,dt,dx,dz,s_u1,s_w1,s_P,s_Q,vp,npml,
                             s_px1,s_px0,s_pz1,s_pz0,s_qx1,s_qx0,s_qz1,s_qz0,
                             acoffx1,acoffx2,acoffz1,acoffz2,delta,epsilon,fs,ds,zs,is,false);
            s_u0=s_u1;   s_w0=s_w1;
            s_px0=s_px1; s_pz0=s_pz1;
            s_qx0=s_qx1; s_qz0=s_qz1;

            if(writeSnap && (it%300==0)) {
                cudaMemcpy(e, s_P, nnz*nnx*sizeof(float), cudaMemcpyDeviceToHost);
                for(int i=npml;i<nnx-npml;i++)
                    for(int j=npml;j<nnz-npml;j++)
                        fwrite(&e[i*nnz+j],4L,1,fpSnap);
            }



            /* receivers wavefield */
            shot_record<<<(nx+511)/512, 512>>>
                        (nnx, nnz, nx, nz, npml, it, nt, g_P, shot_Dev, false);
            shot_record<<<(nx+511)/512, 512>>>
                        (nnx, nnz, nx, nz, npml, it, nt, g_Q, shot_Dev, false);
            update_vel<<<(nnx*nnz+511)/512, 512>>>
                        (nx,nz,nnx,nnz,npml,dt,dx,dz,
                         g_u0,g_w0,g_u1,g_w1,g_P,g_Q,coffx1,coffx2,coffz1,coffz2);
            update_stress<<<(nnx*nnz+511)/512, 512>>>
                            (nx,nz,nnx,nnz,dt,dx,dz,g_u1,g_w1,g_P,g_Q,vp,npml,
                             g_px1,g_px0,g_pz1,g_pz0,g_qx1,g_qx0,g_qz1,g_qz0,
                             acoffx1,acoffx2,acoffz1,acoffz2,
                             delta,epsilon,fs,ds,zs,is,false);
            g_u0=g_u1;   g_w0=g_w1;
            g_px0=g_px1; g_pz0=g_pz1;
            g_qx0=g_qx1; g_qz0=g_qz1;

            if(writeSnap && (it%300==0)) {
                cudaMemcpy(e, g_P, nnz*nnx*sizeof(float), cudaMemcpyDeviceToHost);
                for(int i=npml;i<nnx-npml;i++)
                    for(int j=npml;j<nnz-npml;j++)
                        fwrite(&e[i*nnz+j],4L,1,fpSnap);
            }

            cal_migration<<<(nx*nz+511)/512, 512>>>
                            (nnx, nnz, nz, npml, migration, s_P, g_P);

            Poynting_Adcigs<<<(nx*nz+511)/512, 512>>>
                            (nnz, nx, nz, npml, nangle, dangle, adcigs,
                             s_P, s_Q, s_u0, s_w0, g_P, g_Q, g_u0, g_w0);

        }//it

        time = clock() - time;
        printf(", %f min\n", ((float)time)/60.0/CLOCKS_PER_SEC);

    }//is

    migration_illum<<<(nx*nz+511)/512, 512>>>(nx, nz, npml, migration, illumination);

    adcigs_illum<<<(nx*nz*nangle+511)/512, 512>>>(nx, nz, nangle, dangle, adcigs, illumination);

    /* output multi-shot migration */
    cudaMemcpy(e, migration, nz*nx*sizeof(float), cudaMemcpyDeviceToHost);
    laplace_filter(1,nz,nx,e,d);
    fwrite(d,sizeof(float),nx*nz,fpMigration);

    /* output multi-shot illumination */
    cudaMemcpy(e, illumination, nz*nx*sizeof(float), cudaMemcpyDeviceToHost);
    fwrite(e,sizeof(float),nx*nz,fpIllunmination);

    /* output multi-shot adcigs */
    cudaMemcpy(Atemp, adcigs, nz*nx*nangle*sizeof(float), cudaMemcpyDeviceToHost);
    fwrite(Atemp,sizeof(float),nz*nx*nangle,fpAdcigs);

    /* output adcigs stk migration */
    stk_adcigs(nx,nz,nangle,Atemp,d);
    fwrite(d,sizeof(float),nx*nz,fpStkAdcigs);

    /* output smiled adcigs */
    adcigs_smiled(nx,nz,nangle,dAdcigs,Atemp);
    fwrite(Atemp,sizeof(float),nz*nx/dAdcigs*nangle,fpIntervalAdcigs);


    /* file close */
    if(writeSnap)
        fclose(fpSnap);
    if(readShot)
        fclose(fpObsShot);
    else
        fclose(fpCalShot);
    fclose(fpMigration);
    fclose(fpIllunmination);
    fclose(fpAdcigs);
    fclose(fpStkAdcigs);
    fclose(fpIntervalAdcigs);

    /* device memory free */
    cudaFree(coffx1);    cudaFree(acoffx1);
    cudaFree(coffx2);    cudaFree(acoffx2);
    cudaFree(coffz1);    cudaFree(acoffz1);
    cudaFree(coffz2);    cudaFree(acoffz2);

    cudaFree(s_u0);    cudaFree(s_u1);
    cudaFree(s_w0);    cudaFree(s_w1);

    cudaFree(s_P);    cudaFree(s_Q);

    cudaFree(s_px0);    cudaFree(s_px1);
    cudaFree(s_pz0);    cudaFree(s_pz1);
    cudaFree(s_qx0);    cudaFree(s_qx1);
    cudaFree(s_qz0);    cudaFree(s_qz1);

    cudaFree(g_u0);    cudaFree(g_u1);
    cudaFree(g_w0);    cudaFree(g_w1);

    cudaFree(g_P);    cudaFree(g_Q);

    cudaFree(g_px0);    cudaFree(g_px1);
    cudaFree(g_pz0);    cudaFree(g_pz1);
    cudaFree(g_qx0);    cudaFree(g_qx1);
    cudaFree(g_qz0);    cudaFree(g_qz1);

    cudaFree(shot_Dev);

    cudaFree(P_bndr);    cudaFree(Q_bndr);

    cudaFree(migration);
    cudaFree(illumination);
    cudaFree(adcigs);

    /* host memory free */
    free(v);
    free(e);
    free(d);
    free(shot_Hos);
    free(Atemp);

    print_success("Finish Reverse Time Migration!\n");

}//RTM


/**
 *     MAIN FUNCTION
 *         Author: Rong Tao
 */
int main(int argc,char *argv[])
{
    char msg[1024];
    #pragma message("\n\nNote:                   \
                     \n./a.out kind=1 for FD,  \
                     \n./a.out kind=2 for RTM  \
                     \n")


    /* this "if" for arguments line: ./a.out 1 */
    int kind;
    if(argc == 1 ){
        /* clear screen */
        system("clear");
        print_warning("Check the infomation.\n");
        for(int i=0; note[i] != NULL; i++) {

            fprintf(stderr, "%s",note[i]);
            if(i>13) {
                char ch = getchar();
                if(ch == 'q'){
                    sprintf(msg, "%s exit %s: line %d \n",argv[0],__FILE__,__LINE__);
                    print_success(msg);
                    exit(0);
                } else
                    continue;
            } else {
                fprintf(stderr, "\n");
            }
        }
        //assert(argc-1);
        exit(0);

    }else if(argc >= 2){

        sscanf(argv[1],"kind=%d",&kind);

        if(kind != 1 && kind != 2) {
            sprintf(msg, "Let %s kind=1 for FD. =2 for RTM.\n",argv[0]);
            print_error(msg);
            exit(0);

        }else {

            sprintf(msg, "The %s Starting...\n", argv[0]);
            print_info(msg);
        }
    }

    /* Parameters */
    int nx, nz, nt, wtype, nangle, dangle, dAdcigs;
    int ns, ds, fs, zs, npml;
    float dx, dz, dt, pfac, favg;
    bool readShot, writeSnap;

    clock_t start, end;

    /* file */
    /* these for FD */
    char FNvelocity[90]       = {"waxian_vel_201201201.dat"};
    char FNepsilon[90]        = {"waxian_eps_201201201.dat"};
    char FNdelta[90]          = {"waxian_del_201201201.dat"};
    char FNCalShot[90]        = {"shot.dat"};//shot cal
    char FNSnap[90]           = {"snap.dat"};//snap
    char FNIllumination[90]   = {"illumination.dat"};

    /* these for RTM */
    char FNObsShot[90]        = {"shot_obs.dat"};//shot obs

    char FNMigration[90]      = {"migration.dat"};
    char FNAdcigs[90]         = {"adcigs.dat"};
    char FNStkAdcigs[90]      = {"stkadcigs.dat"};
    char FNIntervalAdcigs[90] = {"smiled_adcigs.dat"};


    wtype=1;/* wavelet: 1,2,3 */
    npml=20;/* pml boundary */

    readShot = false;/* true: read shot;
                       flase: use right shot record */
    writeSnap = false;/* true: write;
                        flase: no write snap */

    nx = 201;
    nz = 201;         favg=35;     pfac=1000.0;

    dx=10.0;
    dz=10.0;

    nt=2001;
    dt=0.001;

    ns=5;
    fs=nx/ns/2;
    ds=nx/ns;
    zs=1;

    nangle=70;
    dangle=1;
    dAdcigs=25;



    start = clock();


    if(kind == 1){
        /* FD */
        print_info("FD starting...\n");
        FD( FNvelocity,
            FNepsilon,
            FNdelta,
            FNCalShot,
            FNSnap,
            FNIllumination,
            wtype,
            npml,
            nx,
            nz,
            dx,
            dz,
            nt,
            dt,
            ns,
            fs,
            ds,
            zs,
            favg,
            pfac,
            writeSnap);

    }else if(kind == 2){
        /* RTM */
        print_info("RTM starting...\n");
        RTM(FNvelocity,
            FNepsilon,
            FNdelta,
            FNObsShot,
            FNCalShot,
            FNSnap,
            FNMigration,
            FNIllumination,
            FNAdcigs,
            FNStkAdcigs,
            FNIntervalAdcigs,
            wtype,
            npml,
            nx,
            nz,
            dx,
            dz,
            nt,
            dt,
            ns,
            fs,
            ds,
            zs,
            favg,
            pfac,
            nangle,
            dangle,
            dAdcigs,
            readShot,
            writeSnap);

    } else { }


    end = clock();

    sprintf(msg, "The %s Finished... \n", argv[0]);
    print_info(msg);

    sprintf(msg, "Total %d shots: %f (min)\n", ns, ((float)(end-start))/60.0/CLOCKS_PER_SEC);
    print_info(msg);
    print_success("Done！ File in:\n");
    system("pwd");

    return 0;

}//end of main