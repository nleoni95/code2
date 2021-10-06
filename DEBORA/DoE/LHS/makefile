LIBLOC = /usr/local/lib
EIGEN = $(LIBLOC)/eigen-3.3.7
STOCH = $(LIBLOC)/StochTk++
BOOST = $(LIBLOC)/boost_1_68_0
ARCH  = OLM


BINDIR = ./

LIBS    =  -lpcb++ -lcub++
LIBDIR	= /opt/local/lib  -L$(STOCH)/lib -L/usr/local/lib
INCLUDES= -I$(EIGEN) -I$(STOCH)/includes -I/usr/local/include 
CFLAGS	=  -O3 '-DCOMPLEX=std::complex<double>' -std=c++11
CPP     = g++
CODE	= lhs.exe
INCLS   = sampler.h

OBJECT= lhs.o

code : $(OBJECT)
	$(CPP) -o $(CODE) $(CFLAGS) $(OBJECT) -L$(LIBDIR) $(LIBS)
 
lhs.o : lhs.cpp
	$(CPP)  $(CFLAGS) $(INCLUDES) -c lhs.cpp

clean:
	rm -f $(OBJECT) $(OBJECTD) $(OBJECTX) *~  core* $(CODE) fort.* *.gnu
