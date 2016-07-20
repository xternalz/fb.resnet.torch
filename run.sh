nGPU=8
runCmd="mpirun "

for (( i=1; i<=$nGPU; i++ ))
do
	runCmd="${runCmd}-np 1 -x CUDA_VISIBLE_DEVICES=$((i-1)) th main.lua -dataset cifar10 -nGPU 1 -batchSize 256 -depth 218 -multiverso true -netType preresnet"
	if [ $i -lt $nGPU ]
	then
		runCmd="${runCmd} : "
	fi
done

eval $runCmd
