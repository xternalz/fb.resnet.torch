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


.PHONY: 1P4G0M0.1LR128B0S 1P8G0M0.1LR128B0S 4P1G1M0.1LR128B0S 4P1G1M0.1LR128B1S 4P1G1M0.05LR128B0S 4P1G1M0.05LR64B1S 8P1G1M0.1LR128B0S 8P1G1M0.1LR128B1S 8P1G1M0.05LR64B0S 8P1G1M0.05LR64B1S

1P4G0M0.1LR128B0S:
	th main.lua -dataset cifar10 -nGPU 4 -batchSize 128 -depth 32
1P8G0M0.1LR128B0S:
	th main.lua -dataset cifar10 -nGPU 8 -batchSize 128 -depth 32

4P1G1M0.1LR128B0S:
	mpirun -np 4 th main.lua -dataset cifar10 -nGPU 1 -batchSize 128 -depth 32 -multiverso true -sync false
4P1G1M0.1LR128B1S:
	mpirun -np 4 th main.lua -dataset cifar10 -nGPU 1 -batchSize 128 -depth 32 -multiverso true -sync true

4P1G1M0.05LR64B0S:
	mpirun -np 4 th main.lua -dataset cifar10 -nGPU 1 -LR 0.05 -batchSize 64 -depth 32 -multiverso true -sync false
4P1G1M0.05LR64B1S:
	mpirun -np 4 th main.lua -dataset cifar10 -nGPU 1 -LR 0.05 -batchSize 64 -depth 32 -multiverso true -sync true

8P1G1M0.1LR128B0S:
	mpirun -np 8 th main.lua -dataset cifar10 -nGPU 1 -batchSize 128 -depth 32 -multiverso true -sync false
8P1G1M0.1LR128B1S:
	mpirun -np 8 th main.lua -dataset cifar10 -nGPU 1 -batchSize 128 -depth 32 -multiverso true -sync true

8P1G1M0.05LR64B0S:
	mpirun -np 8 th main.lua -dataset cifar10 -nGPU 1 -LR 0.05 -batchSize 64 -depth 32 -multiverso true -sync false
8P1G1M0.05LR64B1S:
	mpirun -np 8 th main.lua -dataset cifar10 -nGPU 1 -LR 0.05 -batchSize 64 -depth 32 -multiverso true -sync true
