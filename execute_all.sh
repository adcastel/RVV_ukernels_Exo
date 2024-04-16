#/bin/bash!

#BCAST-NOSWAP
for k in 32 64 128 256 512 1024 2048 4096
do
make && ./test_uk 1 24 1 24 $k 0 1000 > ${HOSTNAME}_BCAST_$k.dat 
done

#BCAST-SWAP
for k in 32 64 128 256 512 1024 2048 4096
do
make SWAPAB=1 && ./test_uk 1 24 1 24 $k 0 1000 > ${HOSTNAME}_BCAST_SWAPAB_$k.dat 
done

#GATHER-NOSWAP
for k in 32 64 128 256 512 1024 2048 4096
do
make GATHER=1 && ./test_uk 1 24 1 24 $k 0 1000 > ${HOSTNAME}_GATHER_$k.dat 
done

#GATHER-SWAP
for k in 32 64 128 256 512 1024 2048 4096
do
make SWAPAB=1 GATHER=1 && ./test_uk 1 24 1 24 $k 0 1000 > ${HOSTNAME}_GATHER_SWAPAB_$k.dat 
done

