# This code simulates multiple random walks, each consisting of a specified number of jumps, and optionally visualizes the results.
#! to see the graph of multiple walks uncomment line 23
#! to see the histogram of the experiments uncomment line 24

import matplotlib.pyplot as plt
import random

JUMPS = 15
EXPERIMENTS = 1000

def random_walk(jumps):
  position = 0
  track = [position]
  for i in range(jumps):
    position += random.choice([-1, 1])
    track.append(position)
  return track,position

results = []
for j in range (EXPERIMENTS):
  results.append(random_walk(JUMPS))

#plt.plot(random_walk(JUMPS)[0])
#plt.plot([result[0] for result in results],marker = 'o')
plt.hist([result[1] for result in results])
plt.show()