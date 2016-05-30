.PHONY: all 1P1G0M 1P2G0M 1P4G0M 1P2G1M 2P1G1M 4P1G1M

1P1G0M:
	th main.lua -dataset cifar10 -nGPU 1 -batchSize 128 -depth 32
1P2G0M:
	th main.lua -dataset cifar10 -nGPU 2 -batchSize 128 -depth 32
1P4G0M:
	th main.lua -dataset cifar10 -nGPU 4 -batchSize 128 -depth 32
1P2G1M:
	mpirun -np 1 th main.lua -dataset cifar10 -nGPU 2 -batchSize 128 -depth 32 -multiverso true
2P1G1M:
	mpirun -np 2 th main.lua -dataset cifar10 -nGPU 1 -batchSize 128 -depth 32 -multiverso true
4P1G1M:
	mpirun -np 4 th main.lua -dataset cifar10 -nGPU 1 -batchSize 128 -depth 32 -multiverso true
