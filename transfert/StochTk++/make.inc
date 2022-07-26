EIGEN = ~/install/eigen-3.4.0
NLOPT = ~/install/include
SELF = ~/install/StochTk++/includes
AR = ar cr
CPP     = c++
CFLAGS  = -O3 -std=c++11 '-DCOMPLEX=std::complex<double>' -I $(EIGEN) -I $(NLOPT) -I $(SELF)
RANLIB  = ranlib
