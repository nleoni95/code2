TARGET = ../../lib/libcub++.a
SRCS = quadrature.cpp
OBJS = $(SRCS:.cpp=.o)
INCLS = cub++.h ../../includes/pcb++.h 
INCLD = ../../includes
RANLIB = ranlib
EIGEN = /opt/local/include/eigen3/
AR = ar cr
CPP = g++
CXXFLAGS  = -O2 '-DCOMPLEX=std::complex<double>' -I$(INCLD) -I$(EIGEN) -fopenmp

$(TARGET): $(OBJS)
	$(AR) $(TARGET) $(OBJS)
	$(RANLIB) $(TARGET)

quadrature.o : quadrature.cpp $(INCLS)
	$(CPP) $(CXXFLAGS) -c quadrature.cpp

clean:
	rm -f $(OBJS) *~ $(TARGET)
