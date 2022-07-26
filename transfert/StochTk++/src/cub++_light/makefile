include make.inc
TARGET = ../../lib/libcub++_light.a
SRCS = quadrature.cpp
OBJS = $(SRCS:.cpp=.o)
INCLS = cub++_light.h 

$(TARGET): $(OBJS)
	$(AR) $(TARGET) $(OBJS)
	$(RANLIB) $(TARGET)

quadrature.o : quadrature.cpp 
	$(CPP) $(CXXFLAGS) -c quadrature.cpp

clean:
	rm -f $(OBJS) *~ $(TARGET)
