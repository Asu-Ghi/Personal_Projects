import numpy as np
import matplotlib.pyplot as plt

# Adjust array depending on bash output for strong scaling study
A = np.array ( ( (1,113.2819),(2,71.7340), (4,56.2396), (8,52.7426) ) )
slow_time = A[0,1]
A[:,1] = slow_time/A[:,1]
plt.rcParams['figure.figsize'] = (7, 7)
plt.rcParams.update({'font.size': 14})
plt.gca().set_aspect('equal')
plt.scatter (np.log2(A[:,0]),np.log2(A[:,1]),color='maroon',s=100,label='10k Spiral Data')
plt.plot ([0,5],[0,5],'--',color='orange',label='Ideal')
plt.title ('OpenMP MiniNet Strong Scaling Study')
plt.xlabel ('Log$_2$ Cores')
plt.ylabel ('Log$_2$ Speedup')
plt.legend()

save_path = '../results/demonstrations/strong_scaling_plot.png' 
plt.savefig(save_path, format='png') 
print(f"Plot saved to {save_path}")
