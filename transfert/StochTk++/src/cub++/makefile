include ../../make.inc
TARGET = ../../lib/libcub++.a
SRCS = quadrature.cpp
OBJS = $(SRCS:.cpp=.o)
INCLS = cub++.h ../../includes/pcb++.h 

$(TARGET): $(OBJS)
	$(AR) $(TARGET) $(OBJS)
	$(RANLIB) $(TARGET)

quadrature.o : quadrature.cpp $(INCLS)
	$(CPP) $(CFLAGS) -c quadrature.cpp

clean:
	rm -f $(OBJS) *~ $(TARGET)
