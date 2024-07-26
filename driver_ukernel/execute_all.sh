#/bin/bash

ini=8
end=40
step=8
mkdir -p out
for mode in "UNROLL";
do
    for swap in 0 1;
      do
      if [ ${swap} -eq 1 ]; then
            mm="loadBA"
          else
            mm="loadAB"
          fi
        for gather in 0 1;
        do
	  if [ ${gather} -eq 1 ]; then
            gg="gather"
          else
            gg="bcast"
          fi
	  for r in 512 1024;
          do
	    file=out/${mode}_${mm}_${gg}_${r}.dat
	    echo "" > ${file}
            for mr in $(seq ${ini} ${step} ${end});
            do
               for nr in $(seq ${ini} ${step} ${end});
               do 
		       make clean
		       make MR=${mr} NR=${nr} MODE=${mode} GATHER=${gather} SWAP=${swap}
		       ./test_uk ${mr} ${mr} ${nr} ${nr} ${r} 0 1000 >> ${file}
	      done
	    done
         done
      done
   done
done

