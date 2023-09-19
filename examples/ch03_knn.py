import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
plt.style.use('book.mplstyle')
import gaud_sty


# Generate random points for training
np.random.seed(13) # to always get the same points
N = 5
x_h = np.random.normal(1.1,0.3,N)
x_e = np.random.normal(1.9,0.4,N)
if gaud_sty.color==True:
    plt.plot(x_h,np.zeros([N,1]), linestyle='', label="maahinen", c='black', marker=gaud_sty.marker3[0])
    plt.plot(x_e,np.zeros([N,1]), linestyle='', label="haltija", c='red', marker=gaud_sty.marker3[1])
else:
    plt.plot(x_h,np.zeros([N,1]), linestyle='', label="maahinen", c=gaud_sty.gray3[0], marker=gaud_sty.marker3[0])
    plt.plot(x_e,np.zeros([N,1]), linestyle='', label="haltija", c=gaud_sty.gray3[1], marker=gaud_sty.marker3[1])
plt.title('Opetusnäytteitä kahdesta luokasta')
plt.legend()
plt.xlabel('pituus [m]')
plt.axis([0.5,2.5,-1.1,+1.1])
if gaud_sty.savefig:
    if gaud_sty.color:
        plt.savefig(gaud_sty.save_dir+'ch03_knn_1_col.png')
    else:
        plt.savefig(gaud_sty.save_dir+'ch03_knn_1.png')
plt.show()

# Generate random points for testing
N_t =  3
x_h_test = np.random.normal(1.1,0.3,N_t) # h as hobit
x_e_test = np.random.normal(1.9,0.4,N_t) # e as elf
if gaud_sty.color==True:
    plt.plot(x_h,np.zeros([N,1]), label="maahinen", linestyle='', c='black', marker=gaud_sty.marker3[0])
    plt.plot(x_e,np.zeros([N,1]), label="haltija", linestyle='', c='red', marker=gaud_sty.marker3[1])
    plt.plot(x_e_test,np.zeros([N_t,1]), label="tuntematon", linestyle='', c='blue', marker=gaud_sty.marker3[2])
    plt.plot(x_h_test,np.zeros([N_t,1]), linestyle='', c='blue', marker=gaud_sty.marker3[2])
else:
    plt.plot(x_h,np.zeros([N,1]), label="maahinen", linestyle='', c=gaud_sty.gray3[0], marker=gaud_sty.marker3[0])
    plt.plot(x_e,np.zeros([N,1]), label="haltija", linestyle='', c=gaud_sty.gray3[1], marker=gaud_sty.marker3[1])
    plt.plot(x_e_test,np.zeros([N_t,1]), label="tuntematon", linestyle='', c=gaud_sty.gray3[2], marker=gaud_sty.marker3[2])
    plt.plot(x_h_test,np.zeros([N_t,1]), linestyle='', c=gaud_sty.gray3[2], marker=gaud_sty.marker3[2])
plt.title('Luokiteltavia testausnäytteitä')
plt.legend()
plt.xlabel('pituus [m]')
plt.axis([0.5,2.5,-1.1,+1.1])
if gaud_sty.savefig:
    if gaud_sty.color:
        plt.savefig(gaud_sty.save_dir+'ch03_knn_2_col.png')
    else:
        plt.savefig(gaud_sty.save_dir+'ch03_knn_2.png')
plt.show()

# 1-NN classifier

# Form the train input and output vectors  (1: hobit, 2: elf)
x_tr = np.concatenate((x_h,x_e))
y_tr = np.concatenate((1*np.ones([x_h.shape[0],1]),2*np.ones([x_e.shape[0],1])))

# Form the test input and output vectors
x_te = np.concatenate((x_h_test,x_e_test))
y_te = np.concatenate((1*np.ones([N_t,1]),2*np.ones([N_t,1])))

#  run 1-NN
y_te_pred = np.empty(y_te.shape)
for test_ind, test in enumerate(x_te):
    min_dist = np.Inf
    for train_ind, train in enumerate(x_tr):
        dist = np.sum((train-test)**2)
        if dist < min_dist:
            y_te_pred[test_ind] = y_tr[train_ind]
            min_dist = dist

tot_correct = len(np.where(y_te-y_te_pred == 0)[0])
print(f'Classication accuracy: {tot_correct/len(y_te)*100:.2f}%')

if gaud_sty.color==True:
    plt.plot(x_h,np.zeros([N,1]), label="maahinen", linestyle='', c='black', marker=gaud_sty.marker3[0])
    plt.plot(x_e,np.zeros([N,1]), label="haltija", linestyle='', c='red', marker=gaud_sty.marker3[1])
else:
    plt.plot(x_h,np.zeros([N,1]), label="maahinen", linestyle='', c=gaud_sty.gray3[0], marker=gaud_sty.marker3[0])
    plt.plot(x_e,np.zeros([N,1]), label="haltija", linestyle='', c=gaud_sty.gray3[1], marker=gaud_sty.marker3[1])
for s_ind, s in enumerate(x_te):
    if y_te_pred[s_ind] == 1:
        if gaud_sty.color==True:
            plt.plot(s,0,linewidth=1, markersize=12, linestyle='', c='blue', marker=gaud_sty.marker3[0])
        else:
            plt.plot(s,0,linewidth=1, markersize=12, linestyle='', c=gaud_sty.gray3[2], marker=gaud_sty.marker3[0])
    else:
        if gaud_sty.color==True:
            plt.plot(s,0,linewidth=1, markersize=12, linestyle='', c='blue', marker=gaud_sty.marker3[1])
        else:
            plt.plot(s,0,linewidth=1, markersize=12, linestyle='', c=gaud_sty.gray3[2], marker=gaud_sty.marker3[1])
    if y_te_pred[s_ind] != y_te[s_ind]:
        plt.plot(s,0, 'kx', markersize=30)
        #plt.annotate("V", (s, 0.5))
    
plt.title('Luokittelutulos (lähimmän naapurin -menetelmä)')
plt.legend()
plt.xlabel('pituus [m]')
plt.axis([0.5,2.5,-1.1,+1.1])
if gaud_sty.savefig:
    if gaud_sty.color:
        plt.savefig(gaud_sty.save_dir+'ch03_knn_3_col.png')
    else:
        plt.savefig(gaud_sty.save_dir+'ch03_knn_3.png')
plt.show()

np.savetxt("ch03_knn_x_train.txt", x_tr)
np.savetxt("ch03_knn_y_train.txt", y_tr)
np.savetxt("ch03_knn_x_test.txt", x_te)
np.savetxt("ch03_knn_y_test.txt", y_te)
