from collections import deque
memory=deque(maxlen=10000); memory.append((0,1,0.0,1,False)); print("Replay memory size:",len(memory)); print("Sample transition:",memory[0])
