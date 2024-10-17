#/bin/bash

mode=OPT
for ARCH in RVV
do
  for bits in 128 256;
  do
    export RVV_BITS=${bits}
    for prec in 32 16; 
    do 
      for gather in 0 1 2; 
	    do 
	     for swap in 0 1; # faltara el 1 
       do 
          if  [ ${gather} -eq 2 ]; then
          	swap=0
          fi
	        if [ $prec = 32 ]; then
       	    if [ $bits = 128 ]; then
	            ini=4
	            end=32
	            step=4
	            lane=4
	            ininr=2
	            endnr=32
	            stepnr=2
	          else
	            ini=8
	            end=48
	            step=8
	            lane=8
	            ininr=2
	            endnr=32
	            stepnr=2
	          fi
	        else #fp16
       	    if [ $bits = 128 ]; then
	            ini=8
	            end=64
	            step=8
	            lane=8
	            ininr=2
	            endnr=64
	            stepnr=2
	     	    else
	            ini=16
	            end=96
	            step=16
	            lane=16
	            ininr=2
	            endnr=64
	            stepnr=2
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
            if [ ${gather} -eq 0 ]; then
	            gg="bcast"
            else
              gg="macc"
            fi
	        fi 
          mkdir -p kernels
	        for mr in $(seq ${ini} ${step} ${end});
          do
	          for nr in $(seq ${ininr} ${stepnr} ${endnr});
	          do	    
	            ff=kernels_${ARCH}_${mr}x${nr}_fp${prec}
	            dest=kernels/${ARCH}_${bits}_BASE/fp${prec}/${mr}x${nr}/${ss}/${gg}
	            destldX=kernels/${ARCH}_${bits}_LDX/fp${prec}/${mr}x${nr}/${ss}/${gg}
	            destopt=kernels/${ARCH}_${bits}_OPT/fp${prec}/${mr}x${nr}/${ss}/${gg}
              mkdir -p ${destldX}
              mkdir -p ${destopt}
              
              if ! test -f ${dest}/${ff}.c; then
	              echo "${mr} ${nr} ${lane} ${prec} ${swap} ${gather} 1 60 | exocc -o ${dest} --stem ${ff} RVV_generator_macc.py" 
	              echo "${mr} ${nr} ${lane} ${prec} ${swap} ${gather} 1 60" | exocc -o ${dest} --stem ${ff} RVV_generator_macc.py
	            else
	              echo "${dest}/${ff} already exists"
	            fi
	            if test -f ${dest}/${ff}.c; then
	                echo "python3 exo_to_opt_converter.py ${dest}/${ff}.c ${destldX}/${ff}.c 0 ${mr} ${nr} ${prec} ${ARCH}"
	                python3 exo_to_opt_converter.py ${dest}/${ff}.c ${destldX}/${ff}.c 0 ${mr} ${nr} ${prec} ${ARCH}
	                echo "python3 exo_to_opt_converter.py ${dest}/${ff}.h ${destldX}/${ff}.h 0 ${mr} ${nr} ${prec} ${ARCH}"
	                python3 exo_to_opt_converter.py ${dest}/${ff}.h ${destldX}/${ff}.h 0 ${mr} ${nr} ${prec} ${ARCH}
	                echo "python3 exo_to_opt_converter.py ${dest}/${ff}.c ${destopt}/${ff}.c 1 ${mr} ${nr} ${prec} ${ARCH}"
	                python3 exo_to_opt_converter.py ${dest}/${ff}.c ${destopt}/${ff}.c 1 ${mr} ${nr} ${prec} ${ARCH}
	                echo "python3 exo_to_opt_converter.py ${dest}/${ff}.h ${destopt}/${ff}.h 1 ${mr} ${nr} ${prec} ${ARCH}"
	                python3 exo_to_opt_converter.py ${dest}/${ff}.h ${destopt}/${ff}.h 1 ${mr} ${nr} ${prec} ${ARCH}
	            else
	              echo "${mr}x${nr} has not been build"
	            fi
              echo "python3 generate_matrix_base.py ${mr} ${nr} ${lane} ${ARCH} fp${prec} fp${prec} fp${prec} ${swap} ${gather}"
	            python3 generate_matrix_base.py ${mr} ${nr} ${lane} ${ARCH} fp${prec} fp${prec} fp${prec} ${swap} ${gather}
	            echo "python3 generate_matrix.py ${mr} ${nr} ${lane} ${ARCH} fp${prec} fp${prec} fp${prec} ${swap} ${gather} LDX"
	            python3 generate_matrix.py ${mr} ${nr} ${lane} ${ARCH} fp${prec} fp${prec} fp${prec} ${swap} ${gather} LDX
	            echo "python3 generate_matrix.py ${mr} ${nr} ${lane} ${ARCH} fp${prec} fp${prec} fp${prec} ${swap} ${gather} OPT"
	            python3 generate_matrix.py ${mr} ${nr} ${lane} ${ARCH} fp${prec} fp${prec} fp${prec} ${swap} ${gather} OPT
              
              if [ ${gather} -eq 2 ]; then #queremos sustituir lo que no se ha generado bien con el macc
                  mkdir -p kernels/${ARCH}_${bits}_BASE/fp${prec}/${mr}x${nr}/loadBA
                  mkdir -p kernels/${ARCH}_${bits}_LDX/fp${prec}/${mr}x${nr}/loadBA
                  mkdir -p kernels/${ARCH}_${bits}_OPT/fp${prec}/${mr}x${nr}/loadBA
                  cp -r ${dest} kernels/${ARCH}_${bits}_BASE/fp${prec}/${mr}x${nr}/loadBA
                  cp -r ${destldX} kernels/${ARCH}_${bits}_LDX/fp${prec}/${mr}x${nr}/loadBA
                  cp -r ${destopt} kernels/${ARCH}_${bits}_OPT/fp${prec}/${mr}x${nr}/loadBA
              fi
            done;
	        done;
        done;
      done;
    done;
  done;
done;
