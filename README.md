# RVV_ukernels_Exo

This is an Exo-based generator of General Matrix Multiplication Micro-kernels for RISC-V devices.

## Files
- README.md: is this file
- RVV_generator.py: is the Exo-based generator itself.
- exo_to_opt_converter.py: A Python script that removes Exo structs from the C code.
- generate_matrix(_base).py: A Python script that generates a matrix of function pointers.
- generate_uk.sh: A Bash script that calls the overall process.

## How to use
The generator can be used in two manners:
### Solo mode
``echo "MR NR LANE PREC SWAP GATHER UNROLL REGS" | exocc -o DEST --stem FILE``

Where:
- MR is the first dimension of the micro-kernel.
- NR is the second dimension of the micro-kernel.
- LANE is the number of elements that fit into a vectorial register.
- PREC [16|32] is the precision of each element.
- SWAP [0|1] indicates if the loads of A and B are swapped. 0 False, 1 True.
- GATHER [0|1|2] indicates the method for loading B to the micro-kernel. 0 for Broadcast, 1 for Gather, 2 for Direct.
- UNROLL indicates the factor for the k-loop unroll
- REGS indicates the limit of registers for the generation. If the requested micro-kernel uses more than this number of registers, the execution is aborted.
- DEST is the directory where the micro-kernels are created
- FILE is the name that the micro-kernel file takes. 
  
The following example generates a 12(MR) x 8(NR) microkernel with 4(LANE) elements of fp32(PREC) per register. It loads B before A (SWAP=1) using the Gather approach (GATHER=1). 
The UNROLL is one and uses 32 REGS. The destination (DEST) of the file is the `test` folder, and the file name (FILE) is `microkernel`.

``echo "12 8 4 32 1 1 1 32" | exocc -o test --stem microkernel`` 

#### Optimizing the C code
The output of the generator includes several Exo structs. We can remove them using the following script:

``python3 exo_to_opt_converter.py DEST/FILE.c DEST/FILE.c VERSION MR NR fpPREC ARCH``

``python3 exo_to_opt_converter.py DEST/FILE.h DEST/FILE.h VERSION MR NR fpPREC ARCH``

MR, NR, DEST, FILE, and PREC are the same as in the generator.
- VERSION [0|1] indicates if we want to add MR and NR to the memory access. 0 indicates that leading dimensions are used, and 1 MR and NR.
- ARCH indicates the architecture. Only RVV is supported at the moment.

#### Generating the matrix
This Python script generates a matrix of function pointers for ease of use of the micro-kernel set from applications.

``python3 generate_matrix_base.py MR NR LANE ARCH fp$PREC fpPREC fpPREC SWAP GATHER``

MR, NR, LANE, ARCH, SWAP GATHER, and PREC are the same as previously.

If we have optimized the code using the previous step, we can generate the matrix as follows:

``python3 generate_matrix.py MR NR LANE ARCH fp$PREC fpPREC fpPREC SWAP GATHER MODE``

MODE takes the OPT or LDX value depending on the VERSION selected in the optimization step.

### All in one

Using the ``generate_uk.sh`` script will take care of all the abovementioned steps and will generate the entire set of micro-kernels. 
