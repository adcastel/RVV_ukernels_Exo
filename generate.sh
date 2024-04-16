#/bin/bash

for pr in 32; 
do 
	for swap in 0 1; 
	do  
		for gather in 0 1; 
		do  
			#if ! test -f kernels/kernel_${i}x${j}_b${b}.c ; then
			if [ ${swap} -eq 1 ]; then
				ss="swapAB_"
			else
				ss=""
			fi    
			if [ ${gather} -eq 1 ]; then
				gg="gather"
			else
				gg="bcast"
			fi    
			if [ ${pr} -eq 16 ]; then
				ll=8
			else
				ll=4
			fi    
			if ! test -f pruebas/kernels_rvv_${gg}_${ss}fp${pr}.c ; then
			    echo "25 25 ${ll} ${pr} ${swap} ${gather}" | exocc -o pruebas --stem kernels_rvv_${gg}_${ss}fp${pr} RVV_generator.py 
			fi
 		done; 
	done; 
done
