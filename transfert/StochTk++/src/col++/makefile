TARGET = ../../lib/libcol++.a
SRCS = interpolation.cpp
OBJS = $(SRCS:.cpp=.o)
INCLS = col++.h ../../includes/pcb++.h ../../lib/libpcb++.a

$(TARGET): $(OBJS)
	$(AR) $(TARGET) $(OBJS)
	$(RANLIB) $(TARGET)

interpolation.o : interpolation.cpp $(INCLS)
	$(CPP) $(CFLAGS) -c interpolation.cpp

clean:
	rm -f $(OBJS) *~ $(TARGET)
