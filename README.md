To compile GPU version, cd gpu

make
./a.out number_of_particle
python test.py

Test.py will output the absolute per element error of GPU and CPU 
version


To compile OpenMP version, cd openmp


make
./a.out number_of_particle number_of_threads
python test.py

Test.py will output the absolute per element error of serial and 
static,dynamic,guided version. 
