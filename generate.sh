#/bin/bash

mode=OPT
for ARCH in RVV
do
  for bits in 256 128;
  do
    export RVV_BITS=${bits}
    for prec in 32 16; 
    do 
      for swap in 0 1; 
      do  
        for gather in 0 1; 
	do 
	  if [ $prec = 32 ]; then
       	    if [ $bits = 128 ]; then
	      ini=4
	      end=24
	      step=4
	      lane=4
	      ininr=4
	      endnr=24
	      stepnr=4
	    else
	      ini=8
	      end=48
	      step=8
	      lane=8
	      ininr=4
	      endnr=48
	      stepnr=4
	   fi
	 else #fp16
       	   if [ $bits = 128 ]; then
	     ini=8
	     end=48
	     step=8
	     lane=8
	     ininr=4
	     endnr=48
	     stepnr=4
	     export RVV_BITS=128
	  else
	     ini=16
	     ininr=4
	     end=96
	     step=16
	     lane=16
	     ininr=4
	     endnr=96
	     stepnr=4
	     export RVV_BITS=256
	  fi
	fi
	if [ ${swap} -eq 1 ]; then
	  ss="loadBA"
	else
	  ss="loadAB"
	fi    
	if [ ${gather} -eq 1 ]; then
	  gg="gather"
	else
	  gg="bcast"
	fi 
        mkdir -p kernels
	for mr in $(seq ${ini} ${step} ${end});
        do
	  for nr in $(seq ${ininr} ${stepnr} ${endnr});
	  do	    
	    ff=kernels_${ARCH}_${mr}x${nr}_fp${prec}
	    dest=kernels/${ARCH}_${bits}_${mode}/fp${prec}/${mr}x${nr}/${ss}/${gg}
            if ! test -f ${dest}/${ff}.c; then
	      echo "${mr} ${nr} ${lane} ${prec} ${swap} ${gather} 60 | exocc -o ${dest} --stem ${ff} RVV_generator.py" 
              echo "${mr} ${nr} ${lane} ${prec} ${swap} ${gather} 60" | exocc -o ${dest} --stem ${ff} RVV_generator.py
	      if test -f ${dest}/${ff}.c; then
	        echo "python3 exo_to_opt_converter.py ${dest}/${ff}.c ${dest}/${ff}.c 1 ${mr} ${nr} ${prec} ${ARCH}"
	        python3 exo_to_opt_converter.py ${dest}/${ff}.c ${dest}/${ff}.c 1 ${mr} ${nr} ${prec} ${ARCH}
	        echo "python3 exo_to_opt_converter.py ${dest}/${ff}.h ${dest}/${ff}.h 1 ${mr} ${nr} ${prec} ${ARCH}"
	        python3 exo_to_opt_converter.py ${dest}/${ff}.h ${dest}/${ff}.h 1 ${mr} ${nr} ${prec} ${ARCH}
	      else
	          echo "${mr}x${nr} has not been build"
	      fi
           else
	      echo "${dest}/${ff} already exists"
	   fi
	   echo "python3 generate_matrix.py ${mr} ${nr} ${lane} ${ARCH} fp${prec} fp${prec} fp${prec} ${swap} ${gather}"
	   python3 generate_matrix.py ${mr} ${nr} ${lane} ${ARCH} fp${prec} fp${prec} fp${prec} ${swap} ${gather} ${mode}
          done;
	done;
      done;
     done;
    done;
  done;
done;
