CC = g++
CFLAGS = -g -Wall -Ofast
SRCS = detection.cpp classification.cpp extraction.cpp utils.cpp main.cpp
PROG = main
 
OPENCV = `pkg-config opencv --cflags --libs`
LIBS = $(OPENCV)
 
$(PROG):$(SRCS)
	$(CC) $(CFLAGS) -o $(PROG) $(SRCS) $(LIBS)
