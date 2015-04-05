#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include "gputimer.h"
#define BLOCK_SIZE (1024)
double get_walltime()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double) (tp.tv_sec + tp.tv_usec*1e-6);
}
void Write(float* buffer, int np, char* output){
   FILE *f;
   f=fopen(output,"w");
    for(int i=0;i<np;++i){
       fprintf(f,"%15f \n",buffer[i]);
    }
    fclose(f);
}
void force_repulsion(int np, const float *pos, float L, float krepulsion, 
    float *forces)
{
    int i, j;
    float posi[4];
    float rvec[4];
    float s2, s, f;

    // initialize forces to zero
    for (i=0; i<3*np; i++)
        forces[i] = 0.;

    // loop over all pairs
    for (i=0; i<np; i++)
    {
        posi[0] = pos[3*i  ];
        posi[1] = pos[3*i+1];
        posi[2] = pos[3*i+2];

        for (j=i+1; j<np; j++)
        {
            // compute minimum image difference
            rvec[0] = remainder(posi[0] - pos[3*j  ], L);
            rvec[1] = remainder(posi[1] - pos[3*j+1], L);
            rvec[2] = remainder(posi[2] - pos[3*j+2], L);

            s2 = rvec[0]*rvec[0] + rvec[1]*rvec[1] + rvec[2]*rvec[2];

            if (s2 < 4)
            {
                s = sqrt(s2);
                rvec[0] /= s;
                rvec[1] /= s;
                rvec[2] /= s;
                f = krepulsion*(2.-s);

                forces[3*i  ] +=  f*rvec[0];
                forces[3*i+1] +=  f*rvec[1];
                forces[3*i+2] +=  f*rvec[2];
                forces[3*j  ] += -f*rvec[0];
                forces[3*j+1] += -f*rvec[1];
                forces[3*j+2] += -f*rvec[2];
            }
        }
    }
}

__global__ void gpu_find_repulsion(int np, float*pos, float L, float krepulsion, float* forces){
     int i = blockDim.x * blockIdx.x + threadIdx.x;
     if(i<np){
     	int j;
     	float posi[3];
     	float rvec[3];
     	float s2, s, f;
     	posi[0] = pos[3*i  ];
     	posi[1] = pos[3*i+1];
     	posi[2] = pos[3*i+2];
	 	for (j=i+1; j<np; ++j){
        // compute minimum image difference
         	rvec[0] = remainder(posi[0] - pos[3*j  ], L);
         	rvec[1] = remainder(posi[1] - pos[3*j+1], L);
         	rvec[2] = remainder(posi[2] - pos[3*j+2], L);
         	s2 = rvec[0]*rvec[0] + rvec[1]*rvec[1] + rvec[2]*rvec[2];
         	if (s2 < 4){
             	s = sqrt(s2);
             	rvec[0] /= s;
             	rvec[1] /= s;
             	rvec[2] /= s;
             	f = krepulsion*(2.-s);
             	forces[3*i  ] +=  f*rvec[0];
             	forces[3*i+1] +=  f*rvec[1];
                forces[3*i+2] +=  f*rvec[2];
             	atomicAdd(&forces[3*j],-f*rvec[0]);
             	atomicAdd(&forces[3*j+1],-f*rvec[1]);
             	atomicAdd(&forces[3*j+2],-f*rvec[2]);
            }
        }
     }
}


int main(int argc, char *argv[])
{
    int i;
    int np = 100;             // default number of particles
    float phi = 0.3;         // volume fraction
    float krepulsion = 125.; // force constant
    float *pos;
    float *forces;
    double time0, time1;

    if (argc > 1)
        np = atoi(argv[1]);

    // compute simulation box width
    float L = pow(4./3.*3.1415926536*np/phi, 1./3.);

    // generate random particle positions inside simulation box
    forces = (float *) malloc(3*np*sizeof(float));
    pos    = (float *) malloc(3*np*sizeof(float));
    for (i=0; i<3*np; i++)
        pos[i] = rand()/(float)RAND_MAX*L;
    time0 = get_walltime();
    force_repulsion(np, pos, L, krepulsion, forces);
    time1 = get_walltime();
    //print performance and write to file
    printf("number of particles: %d\n", np);
    printf("elapsed time of cpu program: %f seconds\n", time1-time0);
    Write(forces,3*np,"cpu_output"); 
    //reinitialization of forces
    // for(int i=0;i<np*3;++i) forces[i]=0.;
    //gpu program
    float *gpu_pos;
    float *gpu_forces;
    int bytes=3*np*sizeof(float);
    GpuTimer timer;
    cudaMalloc((void**)&gpu_pos,bytes);
    cudaMalloc((void**)&gpu_forces,bytes);
    cudaMemcpy(gpu_pos, pos, bytes, cudaMemcpyHostToDevice);
    cudaMemset(gpu_forces, 0, bytes);
    timer.Start();
    gpu_find_repulsion<<<(3*np+BLOCK_SIZE-1)/BLOCK_SIZE,BLOCK_SIZE>>>(np, gpu_pos, L, krepulsion, gpu_forces);
    timer.Stop();
    cudaMemcpy(forces, gpu_forces, bytes, cudaMemcpyDeviceToHost);
    printf("number of particles: %d\n", np);
    printf("elapsed time of gpu program: %f seconds\n", timer.Elapsed()/1000);
    Write(forces,3*np,"gpu_output");
    printf("speed up of gpu is %f \n",(time1-time0)/(timer.Elapsed()/1000));
    cudaFree(gpu_pos);
    cudaFree(gpu_forces);
    free(forces);
    free(pos);

    return 0;
}
