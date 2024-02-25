#/bin/bash

for b in 0 1; 
do 
	for i in $(seq 1 1 24); 
	do 
		for j in $(seq 1 1 24); 
		do  
			if ! test -f kernels/kernel_${i}x${j}_b${b}.c ; then
			    echo "$i $j 4 $b" | exocc -o kernels --stem kernel_${i}x${j}_b${b} NEON_generator.py; 
                        fi
    	       done; 
	done; 
done
