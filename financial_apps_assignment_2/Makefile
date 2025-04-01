CC = gcc
CFLAGS = -Wall -Wextra
LDFLAGS = -lgsl -lgslcblas -lm

.PHONY: all clean

all: mcpricer parallel_mcpricer

mcpricer: mcpricer.c
	$(CC) $(CFLAGS) mcpricer.c -o mcpricer $(LDFLAGS)

parallel_mcpricer: parallel_mcpricer.c
	$(CC) $(CFLAGS) -fopenmp parallel_mcpricer.c -o parallel_mcpricer $(LDFLAGS)

clean:
	rm -f mcpricer parallel_mcpricer
