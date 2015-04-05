#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>
double get_walltime()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double) (tp.tv_sec + tp.tv_usec*1e-6);
}

void force_repulsion(int np, const double *pos, double L, double krepulsion, 
    double *forces)
{   int i, j;
    double posi[4];
    double rvec[4];
    double s2, s, f;

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

void static_parallel_forces(int np, const double *pos, double L, double krepulsion, 
    double *forces, int threads){  
    int i,j;
    #pragma omp parallel num_threads(threads) private(i,j)
    {
        #pragma omp for schedule(static)
        for (i=0; i<np; i++){
            for (j=i+1; j<np; j++){
            // compute minimum image difference
                double rvec0 = remainder(pos[3*i  ] - pos[3*j  ], L);
                double rvec1 = remainder(pos[3*i+1] - pos[3*j+1], L);
                double rvec2 = remainder(pos[3*i+2] - pos[3*j+2], L);
                double s2 = rvec0*rvec0 + rvec1*rvec1 + rvec2*rvec2;
                if (s2 < 4){
                    double s = sqrt(s2);
                    rvec0 /= s;
                    rvec1 /= s;
                    rvec2 /= s;
                    double f = krepulsion*(2.-s);
                    forces[3*i  ] +=  f*rvec0;
                    forces[3*i+1] +=  f*rvec1;
                    forces[3*i+2] +=  f*rvec2;
                    #pragma omp atomic
                    forces[3*j  ] += -f*rvec0;
                    #pragma omp atomic
                    forces[3*j+1] += -f*rvec1;
                    #pragma omp atomic
                    forces[3*j+2] += -f*rvec2;
                }
            }
        }
    }
}


void dynamic_parallel_forces(int np, const double *pos, double L, double krepulsion, 
    double *forces, int threads){  
    int i,j;
    #pragma omp parallel num_threads(threads) private(i,j)
    {
        #pragma omp for schedule(dynamic)
        for (i=0; i<np; i++){
            for (j=i+1; j<np; j++){
            // compute minimum image difference
                double rvec0 = remainder(pos[3*i  ] - pos[3*j  ], L);
                double rvec1 = remainder(pos[3*i+1] - pos[3*j+1], L);
                double rvec2 = remainder(pos[3*i+2] - pos[3*j+2], L);
                double s2 = rvec0*rvec0 + rvec1*rvec1 + rvec2*rvec2;
                if (s2 < 4){
                    double s = sqrt(s2);
                    rvec0 /= s;
                    rvec1 /= s;
                    rvec2 /= s;
                    double f = krepulsion*(2.-s);
                    forces[3*i  ] +=  f*rvec0;
                    forces[3*i+1] +=  f*rvec1;
                    forces[3*i+2] +=  f*rvec2;
                    #pragma omp atomic
                    forces[3*j  ] += -f*rvec0;
                    #pragma omp atomic
                    forces[3*j+1] += -f*rvec1;
                    #pragma omp atomic
                    forces[3*j+2] += -f*rvec2;
                }
            }
        }
    }
}



void guided_parallel_forces(int np, const double *pos, double L, double krepulsion, 
    double *forces, int threads){  
    int i,j;
    #pragma omp parallel num_threads(threads) private(i,j)
    {
        #pragma omp for schedule(guided)
        for (i=0; i<np; i++){
            for (j=i+1; j<np; j++){
            // compute minimum image difference
                double rvec0 = remainder(pos[3*i  ] - pos[3*j  ], L);
                double rvec1 = remainder(pos[3*i+1] - pos[3*j+1], L);
                double rvec2 = remainder(pos[3*i+2] - pos[3*j+2], L);
                double s2 = rvec0*rvec0 + rvec1*rvec1 + rvec2*rvec2;
                if (s2 < 4){
                    double s = sqrt(s2);
                    rvec0 /= s;
                    rvec1 /= s;
                    rvec2 /= s;
                    double f = krepulsion*(2.-s);
                    forces[3*i  ] +=  f*rvec0;
                    forces[3*i+1] +=  f*rvec1;
                    forces[3*i+2] +=  f*rvec2;
                    #pragma omp atomic
                    forces[3*j  ] += -f*rvec0;
                    #pragma omp atomic
                    forces[3*j+1] += -f*rvec1;
                    #pragma omp atomic
                    forces[3*j+2] += -f*rvec2;
                }
            }
        }
    }
}

void Write(double* buffer, int np, char* output){
    FILE *f;
    f=fopen(output,"w");
    for(int i=0;i<np;++i){
       fprintf(f,"%15f \n",buffer[i]);
    }
    fclose(f);
}

int main(int argc, char *argv[])
{
    int i;
    int np = 100;             // default number of particles
    double phi = 0.3;         // volume fraction
    double krepulsion = 125.; // force constant
    double *pos;
    double *forces;
    double *p_forces;
    double time0, time1;
    double time2,time3;
    int num_threads=1;
    if (argc > 2)
    {  np = atoi(argv[1]); num_threads=atoi(argv[2]);}
    // compute simulation box width
    double L = pow(4./3.*3.1415926536*np/phi, 1./3.);

    // generate random particle positions inside simulation box
    forces = (double *) malloc(3*np*sizeof(double));
    pos    = (double *) malloc(3*np*sizeof(double));
    for (i=0; i<3*np; i++)
        pos[i] = rand()/(double)RAND_MAX*L;
   
    // measure execution time of this function

    time0 = get_walltime();
    force_repulsion(np, pos, L, krepulsion, forces);
    time1 = get_walltime();
    printf("number of particles: %d\n", np);
    printf("elapsed time of sequential program: %f\n", time1-time0);
    Write(forces,3*np,"serial_output");
    //

    p_forces= (double *) malloc(3*np*sizeof(double));
    for(int i=0;i<3*np;++i) p_forces[i]=0;
    time2 =  get_walltime();
    static_parallel_forces(np, pos, L, krepulsion, p_forces,num_threads);
    time3 =  get_walltime();
    //printf("number of particles: %d\n", np);
    printf("elapsed time of openmp program with static scheduling is : %f\n", time3-time2);
    Write(p_forces,3*np,"static_parallel_output");


    for(int i=0;i<3*np;++i) p_forces[i]=0;
    time2 =  get_walltime();
    guided_parallel_forces(np, pos, L, krepulsion, p_forces,num_threads);
    time3 =  get_walltime();
    //printf("number of particles: %d\n", np);
    printf("elapsed time of openmp program with guided scheduling is : %f\n", time3-time2);
    Write(p_forces,3*np,"dynamic_parallel_output");



    for(int i=0;i<3*np;++i) p_forces[i]=0;
    time2 =  get_walltime();
    dynamic_parallel_forces(np, pos, L, krepulsion, p_forces,num_threads);
    time3 =  get_walltime();
    //printf("number of particles: %d\n", np);
    printf("elapsed time of openmp program with dynamic scheduling : %f\n", time3-time2);
    Write(p_forces,3*np,"guided_parallel_output");








    //printf("speed up using %d threads using is %f \n", num_threads,(time1-time0)/(time3-time2) );
    free(forces);
    free(pos);
    free(p_forces);
    return 0;
}
