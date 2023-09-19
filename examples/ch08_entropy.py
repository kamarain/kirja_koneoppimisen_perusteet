import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from numpy.linalg import inv

import gaud_sty
plt.style.use('book.mplstyle')

p_pos = np.arange(0,1,0.001)

plt.plot(p_pos,-p_pos*np.log2(p_pos+0.0000001)-(1-p_pos)*np.log2(1-p_pos+0.0000001),'k-')
plt.grid()
plt.xlabel('p_pos')
plt.ylabel('Entropia')
plt.xticks(np.arange(0, 1, step=0.1))
plt.savefig(gaud_sty.save_dir+'ch08_entropy_1.png')
plt.show()
