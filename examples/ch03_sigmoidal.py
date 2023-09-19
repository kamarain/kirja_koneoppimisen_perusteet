import matplotlib.pyplot as plt
import numpy as np
import gaud_sty

# Book stylebook
plt.style.use('book.mplstyle')

# Coodinate system
plt.xlabel('x')
plt.ylabel('y')
#plt.axis([0.5,4.0,-1.1,+1.1])

# Plot logsig
N = 101
x = np.linspace(-3.0,+3.0,N)
if gaud_sty.color==True:
    plt.plot(x, 1/(1+np.exp(-x)), linestyle=gaud_sty.line5[0], label="logsig(x)")
    plt.plot(x, 1/(1+np.exp(-2*x)), linestyle=gaud_sty.line5[1], label="logsig(2x)")
    plt.plot(x, 1/(1+np.exp(-5*x)), linestyle=gaud_sty.line5[2], label="logsig(5x)")
    plt.plot(x, 1/(1+np.exp(-5*x+5)), linestyle=gaud_sty.line5[3], label="logsig(5x-5))")
else:
    plt.plot(x, 1/(1+np.exp(-x)), c=gaud_sty.gray5[0], linestyle=gaud_sty.line5[0], label="logsig(x)")
    plt.plot(x, 1/(1+np.exp(-2*x)), c=gaud_sty.gray5[1], linestyle=gaud_sty.line5[1], label="logsig(2x)")
    plt.plot(x, 1/(1+np.exp(-5*x)), c=gaud_sty.gray5[2], linestyle=gaud_sty.line5[2], label="logsig(5x)")
    plt.plot(x, 1/(1+np.exp(-5*x+5)), c=gaud_sty.gray5[3], linestyle=gaud_sty.line5[3], label="logsig(5x-5))")
plt.title('Logistinen funktio')
plt.legend()
if gaud_sty.savefig:
    if gaud_sty.color:
        plt.savefig(gaud_sty.save_dir+'ch03_sigmoidal_col.png')
    else:
        plt.savefig(gaud_sty.save_dir+'ch03_sigmoidal.png')
plt.show()
