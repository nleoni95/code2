# Makefile for Writing Make Files Example

# *****************************************************
# Variables to control Makefile operation

CC = g++
CFLAGS = -O3 -std=c++11
CPPFLAGS = -I ~/install/include -I ~/install/StochTk++/includes -I ~/install/eigen-3.4.0
LDFLAGS = -L ~/install/StochTk++/lib -L ~/install/lib
LIBS = -lgp++ -lnlopt

# ****************************************************
# Targets needed to bring the executable up to date

main: main.o densities.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $(LDFLAGS) -o main.exe main.o densities.o $(LIBS)

main.o: main.cpp densities.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -c main.cpp

densities.o: densities.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -c densities.cpp

clean:
	rm -f *.o *.exe
