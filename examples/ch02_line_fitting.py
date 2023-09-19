import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
plt.style.use('book.mplstyle')
import gaud_sty



#
# 1. Plot random lines
for foo in range(10):
    a = np.random.uniform(-50.0,+50.0,1)
    b = np.random.uniform(0.0,+150.0,1)
    x = np.linspace(0,2.0,10)
    y = a*x+b
    if gaud_sty.color==True:
        plt.plot(x,y)
    else:
        plt.plot(x,y,c=gaud_sty.gray10[foo])
plt.title('Suoria y=ax+b parametrien a ja b eri arvoilla (x: pituus, y: paino)')
plt.xlabel('pituus [m]')
plt.ylabel('paino [kg]')
plt.axis([0,2,0,150])
if gaud_sty.savefig:
    if gaud_sty.color:
        plt.savefig(gaud_sty.save_dir+'ch02_line_fitting_1_col.png')
    else:
        plt.savefig(gaud_sty.save_dir+'ch02_line_fitting_1.png')
plt.show()

#
# 2. Plot pre-defined lines
plt.clf()
a=0
b=0
y = a*x+b
if gaud_sty.color==True:
    plt.plot(x,y,label='a=0 b=0', linestyle=gaud_sty.line5[0], marker=gaud_sty.marker5[0])
else:
    plt.plot(x,y,label='a=0 b=0', c=gaud_sty.gray10[0], linestyle=gaud_sty.line5[0], marker=gaud_sty.marker5[0])
a=10
b=0
y = a*x+b
if gaud_sty.color==True:
    plt.plot(x,y,label='a=10 b=0', linestyle=gaud_sty.line5[1], marker=gaud_sty.marker5[1])
else:
    plt.plot(x,y,label='a=10 b=0', c=gaud_sty.gray10[1], linestyle=gaud_sty.line5[1], marker=gaud_sty.marker5[1])
a=0
b=40
y = a*x+b
if gaud_sty.color==True:
    plt.plot(x,y,label='a=0 b=40', linestyle=gaud_sty.line5[2], marker=gaud_sty.marker5[2])
else:
    plt.plot(x,y,label='a=0 b=40', c=gaud_sty.gray10[2], linestyle=gaud_sty.line5[2], marker=gaud_sty.marker5[2])
a=10
b=40
y = a*x+b
if gaud_sty.color==True:
    plt.plot(x,y,label='a=10 b=40', linestyle=gaud_sty.line5[3], marker=gaud_sty.marker5[3])
else:
    plt.plot(x,y,label='a=10 b=40', c=gaud_sty.gray10[3], linestyle=gaud_sty.line5[3], marker=gaud_sty.marker5[3])
a=0
b=80
y = a*x+b
if gaud_sty.color==True:
    plt.plot(x,y,label='a=0 b=80', linestyle=gaud_sty.line5[4], marker=gaud_sty.marker5[4])
else:
    plt.plot(x,y,label='a=0 b=80', c=gaud_sty.gray10[4], linestyle=gaud_sty.line5[4], marker=gaud_sty.marker5[4])
plt.legend()
plt.title('Suoria parametrien a ja b eri arvoilla.')
plt.xlabel('pituus [m]')
plt.ylabel('paino [kg]')
plt.axis([0,2,-20,150])
if gaud_sty.savefig:
    if gaud_sty.color:
        plt.savefig(gaud_sty.save_dir+'ch02_line_fitting_2_col.png')
    else:
        plt.savefig(gaud_sty.save_dir+'ch02_line_fitting_2.png')
plt.show()

#
# 3. Plot lines through one given point
plt.clf()
# point 1
x1 = 1.11
y1 = 85.0
if gaud_sty.color==True:
    plt.plot(x1, y1)
else:
    plt.plot(x1, y1, 'ko')
plt.annotate("(1.11,85.0)", (x1, y1), textcoords="offset points",xytext=(-0.1,10),ha='right')
a=0
b=y1-a*1.11
y = a*x+b
if gaud_sty.color==True:
    plt.plot(x,y,label='a=0', linestyle=gaud_sty.line6[0])
else:
    plt.plot(x,y,label='a=0',c=gaud_sty.gray6[0], linestyle=gaud_sty.line6[0])
a=1
b=y1-a*1.11
y = a*x+b
if gaud_sty.color==True:
    plt.plot(x,y,label='a=1', linestyle=gaud_sty.line6[1])
else:
    plt.plot(x,y,label='a=1', c=gaud_sty.gray6[1], linestyle=gaud_sty.line6[1])
a=5
b=y1-a*1.11
y = a*x+b
if gaud_sty.color==True:
    plt.plot(x,y,label='a=5', linestyle=gaud_sty.line6[2])
else:
    plt.plot(x,y,label='a=5', c=gaud_sty.gray6[2], linestyle=gaud_sty.line6[2])
a=10
b=y1-a*1.11
y = a*x+b
if gaud_sty.color==True:
    plt.plot(x,y,label='a=10', linestyle=gaud_sty.line6[3])
else:
    plt.plot(x,y,label='a=10', c=gaud_sty.gray6[3], linestyle=gaud_sty.line6[3])
a=20
b=y1-a*1.11
y = a*x+b
if gaud_sty.color==True:
    plt.plot(x,y,label='a=20', linestyle=gaud_sty.line6[4])
else:
    plt.plot(x,y,label='a=20', c=gaud_sty.gray6[4], linestyle=gaud_sty.line6[4])
a=40
b=y1-a*1.11
x = np.linspace(0,2.0,10)
y = a*x+b
if gaud_sty.color==True:
    plt.plot(x,y,label='a=40', linestyle=gaud_sty.line6[5])
else:
    plt.plot(x,y,label='a=40', c=gaud_sty.gray6[5], linestyle=gaud_sty.line6[5])
plt.legend()
plt.title('Suoria, jotka toteuttavat yhtälön 85=a*1.11+b')
plt.xlabel('pituus [m]')
plt.ylabel('paino [kg]')
plt.axis([0,2,0,150])
if gaud_sty.savefig:
    if gaud_sty.color:
        plt.savefig(gaud_sty.save_dir+'ch02_line_fitting_3_col.png')
    else:
        plt.savefig(gaud_sty.save_dir+'ch02_line_fitting_3.png')
plt.show()

#
# 4. Plot the line through two given points
plt.clf()
# point 1
x1 = 1.11
y1 = 85.0
plt.plot(x1, y1,'ko')
plt.annotate("(1.11,85.0)", (x1, y1), textcoords="offset points",xytext=(-0.1,10),ha='right')
x2 = 1.52
y2 = 110.0
plt.plot(x2, y2,'ko')
plt.annotate("(1.52,110.0)", (x2, y2), textcoords="offset points",xytext=(-0.1,10),ha='right')
a=60.98
b=17.32
x = np.linspace(0,2.0,10)
y = a*x+b
if gaud_sty.color==True:
    plt.plot(x,y,'r-',label='y=60.98x+17.32')
else:
    plt.plot(x,y,'k-',label='y=60.98x+17.32')
plt.legend()
plt.title('Suora kahden tunnetun opetuspisteen kautta')
plt.xlabel('pituus [m]')
plt.ylabel('paino [kg]')
plt.axis([0,2,0,150])
if gaud_sty.savefig:
    if gaud_sty.color:
        plt.savefig(gaud_sty.save_dir+'ch02_line_fitting_4_col.png')
    else:
        plt.savefig(gaud_sty.save_dir+'ch02_line_fitting_4.png')
plt.show()

#
# 5. Plot MSE heat map for N points
np.random.seed(42) # to always get the same points
N = 50 
x = np.random.normal(1.1,0.3,N)
a_gt = 50.0
b_gt = 20.0
y_noise =  np.random.normal(0,8,N) # Measurements from the class 1\n",
y = a_gt*x+b_gt+y_noise
plt.plot(x,y,'ko')
plt.title('Opetusaineisto suoran sovitukseen')
plt.xlabel('pituus [m]')
plt.ylabel('paino [kg]')
plt.axis([0,2,0,150])
plt.savefig(gaud_sty.save_dir+'ch02_line_fitting_5.png')
plt.show()

# Compute MSE heat map for different a and b
MSE_ab = np.empty([11,11])
for ai,a in enumerate((range(0,110,10))):
    for bi, b in enumerate((range(0,110,10))):
        y_hat = a*x+b
        MSE_ab[ai][bi] = np.sum((y-y_hat)**2)/N
        
if gaud_sty.color==True:
    plt.imshow(MSE_ab,extent= [-5,105,-5,105], cmap='hot', norm=LogNorm(), interpolation='nearest')
else:
    plt.imshow(MSE_ab,extent= [-5,105,-5,105], cmap='gray', norm=LogNorm(), interpolation='nearest')
plt.colorbar()
plt.ylabel('a')
plt.xlabel('b')
plt.title('MSE a ja b eri arvoille')
if gaud_sty.savefig:
    if gaud_sty.color:
        plt.savefig(gaud_sty.save_dir+'ch02_line_fitting_6_col.png')
    else:
        plt.savefig(gaud_sty.save_dir+'ch02_line_fitting_6.png')
plt.show()

#
# 6. Plot least-square fit through N points
a = (N*np.sum(x*y)-sum(y)*sum(x))/(N*sum(x*x)-sum(x)*sum(x))
b = (-a*sum(x)+sum(y))/N
y_h = a*x+b
MSE = np.sum((y-y_h)**2)/N

plt.plot(x,y,'ko')
x = np.linspace(0,2.0,10)
if gaud_sty.color==True:
    plt.plot(x,a*x+b,'r-')
else:
    plt.plot(x,a*x+b,'k-')
plt.title(f"Sovitettu suora (a={a:.1f}, b={b:.1f}, MSE={MSE:.1f})")
plt.xlabel('pituus [m]')
plt.ylabel('paino [kg]')
plt.axis([0,2,0,150])
if gaud_sty.savefig:
    if gaud_sty.color:
        plt.savefig(gaud_sty.save_dir+'ch02_line_fitting_7_col.png')
    else:
        plt.savefig(gaud_sty.save_dir+'ch02_line_fitting_7.png')
plt.show()
