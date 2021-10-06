LIBLOC = /volatile/nl255551/lib
EIGEN = $(LIBLOC)/eigen-3.3.7
STOCH = $(LIBLOC)/StochTk++
ARCH  = OLM
BOOST = $(LIBLOC)/boost_1_68_0

BINDIR = ./

LIBS    = -lgp++ -lnlopt -lm -lpcb++ -lcub++
LIBDIR	= $(STOCH)/lib -L/usr/local/lib
INCLUDES= -I$(EIGEN) -I$(STOCH)/includes -I/usr/local/include -I$(BOOST)
CFLAGS	= -O3 '-DCOMPLEX=std::complex<double>' -std=c++11
CPP     = c++
CODE	= Koh.exe
OBJECT= main.o

code : $(OBJECT)
	$(CPP) -o $(CODE) $(CFLAGS) $(OBJECT) -L$(LIBDIR) $(LIBS)
 
main.o : main.cpp
	$(CPP)  $(CFLAGS) $(INCLUDES) -c main.cpp

clean:
	rm -f $(OBJECT) $(OBJECTD) $(OBJECTX) *~  core* $(CODE) fort.* *.gnu
