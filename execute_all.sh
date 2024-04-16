#/bin/bash!

#BCAST-NO SWAP
make && ./test_uk 1 24 1 24 512 0 1000 > ${HOSTNAME}_BCAST.dat 
make SWAPAB=1 && ./test_uk 1 24 1 24 512 0 1000 > ${HOSTNAME}_BCAST_SWAPAB.dat 
make GATHER=1 && ./test_uk 1 24 1 24 512 0 1000 > ${HOSTNAME}_GATHER.dat 
make SWAPAB=1 GATHER=1 && ./test_uk 1 24 1 24 512 0 1000 > ${HOSTNAME}_GATHER_SWAPAB.dat 

