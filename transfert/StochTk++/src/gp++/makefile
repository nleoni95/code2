include ../../make.inc
TARGET = ../../lib/libgp++.a
OBJS = gp++.o my_samplers.o
SRCS= $(SRCS:.o=.c)
INCLS   = gp++.h my_samplers.hpp data++.h tps++.h

$(TARGET): $(OBJS)
	$(AR) $(TARGET) $(OBJS)
	$(RANLIB) $(TARGET)

gp++.o : gp++.cpp $(INCLS)
	$(CPP) $(CFLAGS) -c gp++.cpp

my_samplers.o : my_samplers.cpp $(INCLS)
	$(CPP) $(CFLAGS) -c my_samplers.cpp

clean:
	rm -f $(OBJS) *~ $(TARGET)













