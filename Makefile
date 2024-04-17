
# python3 offline.py --num_timesteps=1000 --num_nodes=500 --num_requests=250 --num_servers=10 --only_last_mile=0 --ub=0
all:
	g++ -O3 batching.cpp -lstdc++
	./a.out
