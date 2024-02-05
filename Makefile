CC=/home/adcastel/opt/riscv/bin/riscv64-unknown-linux-gnu-gcc
#CFLAGS= -O0 -g3 -march=rv64gcv_zfh_xtheadc -mabi=lp64d
CFLAGS= -O3 -march=rv64gcv

OBJECTS := exo_matrix.o kernel_col.o
#OBJECTS := exo_matrix.o

all:main

main: $(OBJECTS)
	$(CC) $(CFLAGS) main.c -o test_uk $(OBJECTS) -I$(PWD) -L$(PWD) 

#exo_matrix.o: kernel_col.o
#	$(CC) $(CFLAGS) -c $*.c kernel_col.o -I$(PWD) -L$(PWD)

.c.o:
	$(CC) $(CFLAGS) -c $*.c -I$(PWD) -L$(PWD)

clean:
	rm *.o test_uk
