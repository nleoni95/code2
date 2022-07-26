include ../../make.inc
TARGET = ../../lib/libmra++.a
OBJS = details.o eval_mra.o
SRCS= $(SRCS:.o=.c)
INCLS   = mra++.h ../../includes/pcb++.h ../../lib/libpcb++.a

$(TARGET): $(OBJS)
	$(AR) $(TARGET) $(OBJS)
	$(RANLIB) $(TARGET)

details.o : details.cpp $(INCLS)
	$(CPP) $(CFLAGS) -c details.cpp

eval_mra.o : eval_mra.cpp $(INCLS)
	$(CPP) $(CFLAGS) -c eval_mra.cpp

clean:
	rm -f $(OBJS) *~ $(TARGET)













