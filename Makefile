CC=/home/adcastel/opt/riscv/bin/riscv64-unknown-linux-gnu-gcc

#CFLAGS= -O3 -march=rv64gc0p7v
CFLAGS= -O3 -march=rv64gcv

KPATH=kernels/RVV/

ifeq ($(SWAPAB), 1)
        ifeq ($(GATHER), 1)
	    kernels=kernels/RVV/kernels_rvv_gather_swapAB_fp32.o
	    BCAS=-DGATHER
	else
	    kernels=kernels/RVV/kernels_rvv_bcast_swapAB_fp32.o
	    BCAS=-DBCAST
	endif
	
	SWAP=-DSWAPAB

else
        ifeq ($(GATHER), 1)
	    kernels=kernels/RVV/kernels_rvv_gather_fp32.o
	    BCAS=-DGATHER
	else
	    kernels=kernels/RVV/kernels_rvv_bcast_fp32.o
	    BCAS=-DBCAST
	endif
	SWAP=
endif

OBJECTS := exo_matrix.o $(kernels)

all:main

main: $(OBJECTS)
	$(CC) $(CFLAGS) main.c -o test_uk $(OBJECTS) -I$(PWD) -I$(PWD)/$(KPATH) -L$(PWD) 

$(kernels): 
	$(CC) $(CFLAGS) -o $(kernels) -c $*.c

.c.o:
	$(CC) $(CFLAGS) -c $*.c $(SWAP) $(BCAS) -I$(PWD)/$(KPATH) -L$(PWD)/$(KPATH) 

clean:
	rm *.o test_uk
