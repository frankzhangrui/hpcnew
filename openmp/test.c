#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>

#define N 2000000

double get_walltime()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double) (tp.tv_sec + tp.tv_usec*1e-6);
}

// specify number of threads as the argument

int main(int argc, char **argv)  
{
  double *a, *b, *c;
  int i;
  double t1, t2;
  int nthreads = atoi(argv[1]);

  a = malloc(N*sizeof(double));
  b = malloc(N*sizeof(double));
  c = malloc(N*sizeof(double));
  
  for (i=0; i<N; i++)
    a[i] = b[i] = (double) i;

  t1 = get_walltime();
  #pragma omp parallel num_threads(nthreads) shared(a,b,c) private(i)
  {
    #pragma omp for schedule(static)
    for (i=0; i<N; i++)
      c[i] = a[i] + b[i];
  }
  t2 = get_walltime();
  printf("time: %f\n", t2-t1);

  free(a);
  free(b);
  free(c);

  return 0;
}