LIBLOC = /usr/local/lib
EIGEN = $(LIBLOC)/eigen-3.3.7
STOCH = $(LIBLOC)/StochTk++
BOOST = $(LIBLOC)/boost_1_68_0
ARCH  = OLM


BINDIR = ./

LIBS    =  -lpcb++ -lcub++
LIBDIR	= /opt/local/lib  -L$(STOCH)/lib -L/usr/local/lib
INCLUDES= -I$(EIGEN) -I$(STOCH)/includes -I/usr/local/include -I$(BOOST)
CFLAGS	=  -O3 '-DCOMPLEX=std::complex<double>' -std=c++11
CPP     = g++
CODE	= Halton_QMC.exe
INCLS   = sampler.h

OBJECT= Halton_QMC.o

code : $(OBJECT)
	$(CPP) -o $(CODE) $(CFLAGS) $(OBJECT) -L$(LIBDIR) $(LIBS)
 
Halton_QMC.o : Halton_QMC.cpp
	$(CPP)  $(CFLAGS) $(INCLUDES) -c Halton_QMC.cpp

clean:
	rm -f $(OBJECT) $(OBJECTD) $(OBJECTX) *~  core* $(CODE) fort.* *.gnu
