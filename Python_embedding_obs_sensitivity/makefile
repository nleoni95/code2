LIBLOC = /usr/local/lib
EIGEN = $(LIBLOC)/eigen-3.3.7
STOCH = $(LIBLOC)/StochTk++

BINDIR = ./

LIBS    = -lgp++ -lnlopt -lpython3.8
LIBDIR	= $(STOCH)/lib -L/usr/local/lib -L/home/nl255551/miniconda3/envs/ravik_env/lib
INCLUDES= -I$(EIGEN) -I$(STOCH)/includes -I/usr/local/include -I/home/nl255551/miniconda3/envs/ravik_env/include/python3.8
CFLAGS	= -O3 '-DCOMPLEX=std::complex<double>' -std=c++11
CPP     = g++
CODE	= koh.exe
OBJECT = main.o densities.o

code: $(OBJECT) densities.o
	$(CPP) -o $(CODE) $(CFLAGS) $(OBJECT) -L$(LIBDIR) $(LIBS)

densities.o: densities.cpp
	$(CPP) $(CFLAGS) $(INCLUDES) -c densities.cpp

main.o: main.cpp densities.o
	$(CPP) $(CFLAGS) $(INCLUDES) -c main.cpp

clean:
	rm -f $(OBJECT) $(OBJECTD) $(OBJECTX) *~  core* $(CODE) fort.* *.gnu