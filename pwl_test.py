import matplotlib.pyplot as plt
import patchworklib as pw

fig, ax1 = plt.subplots()
ax1.plot([1,2,3], [4,5,6])


fig, ax2 = plt.subplots()
ax2.plot([7,8,9], [10, 11, 12])

ax12 = ax1|ax2

ax12.savefig("ax12.png")
