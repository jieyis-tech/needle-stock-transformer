// Project-owned exact attention kernels. FP32, first-order, no dropout/mask.
// One CUDA block owns one query (forward/dQ) or one key (dK/dV).
// No atomics, no N x N allocation; dK/dV recompute probabilities separately.
#define INFINITY __int_as_float(0x7f800000)

extern "C" __global__ void attn_forward(
    const float* Q, const float* K, const float* V, float* O, float* LSE,
    int NQ, int NK, int D, int DV, float scale) {
    int i=blockIdx.x, b=blockIdx.y, t=threadIdx.x, B=blockDim.x;
    __shared__ float red[256], p[256], state[3];
    if(t==0){state[0]=-INFINITY;state[1]=0;state[2]=0;}
    __syncthreads();
    float acc=0;
    for(int j0=0;j0<NK;j0+=B){
        int j=j0+t;
        float s=-INFINITY;
        if(j<NK){s=0;for(int a=0;a<D;a++)s+=Q[(b*NQ+i)*D+a]*K[(b*NK+j)*D+a];s*=scale;}
        red[t]=s;__syncthreads();
        for(int h=B/2;h;h/=2){if(t<h)red[t]=fmaxf(red[t],red[t+h]);__syncthreads();}
        if(t==0){float m=fmaxf(state[0],red[0]);state[1]=expf(state[0]-m);state[0]=m;}
        __syncthreads();
        p[t]=(j<NK)?expf(s-state[0]):0;red[t]=p[t];__syncthreads();
        for(int h=B/2;h;h/=2){if(t<h)red[t]+=red[t+h];__syncthreads();}
        if(t==0)state[2]=state[1]*state[2]+red[0];
        if(t<DV){float val=0;for(int r=0;r<B&&j0+r<NK;r++)val+=p[r]*V[(b*NK+j0+r)*DV+t];acc=state[1]*acc+val;}
        __syncthreads();
    }
    if(t<DV)O[(b*NQ+i)*DV+t]=acc/state[2];
    if(t==0)LSE[b*NQ+i]=state[0]+logf(state[2]);
}

extern "C" __global__ void attn_dq(
    const float* Q,const float* K,const float* V,const float* G,const float* LSE,
    const float* Delta,float* DQ,int NQ,int NK,int D,int DV,float scale){
    int i=blockIdx.x,b=blockIdx.y,t=threadIdx.x,B=blockDim.x;
    __shared__ float ds[256];
    float acc=0;
    for(int j0=0;j0<NK;j0+=B){
        int j=j0+t;float s=0,dp=0;
        if(j<NK){
            for(int a=0;a<D;a++)s+=Q[(b*NQ+i)*D+a]*K[(b*NK+j)*D+a];
            for(int a=0;a<DV;a++)dp+=G[(b*NQ+i)*DV+a]*V[(b*NK+j)*DV+a];
            ds[t]=expf(s*scale-LSE[b*NQ+i])*(dp-Delta[b*NQ+i]);
        }else ds[t]=0;
        __syncthreads();
        if(t<D)for(int r=0;r<B&&j0+r<NK;r++)acc+=ds[r]*K[(b*NK+j0+r)*D+t]*scale;
        __syncthreads();
    }
    if(t<D)DQ[(b*NQ+i)*D+t]=acc;
}

extern "C" __global__ void attn_dkv(
    const float* Q,const float* K,const float* V,const float* G,const float* LSE,
    const float* Delta,float* DK,float* DVout,int NQ,int NK,int D,int DV,float scale){
    int j=blockIdx.x,b=blockIdx.y,t=threadIdx.x,B=blockDim.x;
    __shared__ float ds[256],p[256];
    float ak=0,av=0;
    for(int i0=0;i0<NQ;i0+=B){
        int i=i0+t;float s=0,dp=0;
        if(i<NQ){
            for(int a=0;a<D;a++)s+=Q[(b*NQ+i)*D+a]*K[(b*NK+j)*D+a];
            for(int a=0;a<DV;a++)dp+=G[(b*NQ+i)*DV+a]*V[(b*NK+j)*DV+a];
            p[t]=expf(s*scale-LSE[b*NQ+i]);ds[t]=p[t]*(dp-Delta[b*NQ+i]);
        }else{p[t]=0;ds[t]=0;}
        __syncthreads();
        if(t<D)for(int r=0;r<B&&i0+r<NQ;r++)ak+=ds[r]*Q[(b*NQ+i0+r)*D+t]*scale;
        if(t<DV)for(int r=0;r<B&&i0+r<NQ;r++)av+=p[r]*G[(b*NQ+i0+r)*DV+t];
        __syncthreads();
    }
    if(t<D)DK[(b*NK+j)*D+t]=ak;
    if(t<DV)DVout[(b*NK+j)*DV+t]=av;
}
