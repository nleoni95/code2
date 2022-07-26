include ../../make.inc
TARGET = ../../lib/libpcb++.a
OBJS = base.o basis_derive.o eval_pc.o prod_utils.o gauss.o
SRCS= $(SRCS:.o=.c)
INCLS   = pcb++.h

$(TARGET): $(OBJS)
	$(AR) $(TARGET) $(OBJS)
	$(RANLIB) $(TARGET)

base.o : base.cpp $(INCLS)
	$(CPP) $(CFLAGS) -c base.cpp

eval_pc.o : eval_pc.cpp $(INCLS)
	$(CPP) $(CFLAGS) -c eval_pc.cpp

prod_utils.o : prod_utils.cpp $(INCLS)
	$(CPP) $(CFLAGS) -c prod_utils.cpp

basis_derive.o : basis_derive.cpp $(INCLS)
	$(CPP) $(CFLAGS) -c basis_derive.cpp

gauss.o : gauss.cpp $(INCLS)
	$(CPP) $(CFLAGS) -c gauss.cpp

clean:
	rm -f $(OBJS) *~ $(TARGET)













