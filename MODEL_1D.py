"""
Functions for solving a 1D stochastic equation 
The following naming convention of variables are used.
===== ==========================================================
Name  Description
===== ==========================================================
Nx    The number of mesh cells per process; mesh points are numbered
      from 0 to Nx. 
Ntot  Total  number of mesh cells ; mesh points are numbered
      from 0 to Ntot. 
F     The dimensionless number a*dt/dx**2, which implicitly
      specifies the time step.
T     The stop time for the simulation.
a     Variable coefficient (constant).
L     Length of the domain ([0,L]). L is local in the case of MPI 
x     Mesh points in space.
t     Mesh points in time.
n     Index counter in time.
u     Unknown at current/new time level. u is local in the case of MPI
u_1   u at the previous time level.
dx    Constant mesh spacing in x.
dt    Constant mesh spacing in t.
===== ==========================================================
"""
import sys, time
import random
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import scipy.sparse
import scipy.sparse.linalg
from CALC_ALPHA import *

def solver_FE_simple(model,comm,size,rank, dx, dt, Nx, L, Lglo, F, T):
    """
    Simplest expression of the computational algorithm
    using the Forward Euler method and explicit Python loops.
    For this method F <= 0.5 for stability.
    When solving ED or KPZ equation conditions are more stringent F ~0.05
    """
    import time
    t0 = time.clock()
    print("dt =" + str(dt))
    Nt = int(round(T/float(dt)))
    t = np.linspace(0, T, Nt+1)   # mesh points in time

# We add two cells 0 and N+1 position for periodic boundary conditions 
# and GHOST cells in the case of MPI 
#
#  |____|____|____|____|____|.......|____|____| 
#  | 0  | 1  | 2  | 3  | 4  |.......|_Nx_|Nx+1| total Nx+2 entries  
#  |xxxx|______Nx computational cells____|xxxx| 
#
#     At each time step periodic Boundary Conditions are applied as follows
#      _______________________________
#     |                               |
#     |                               |
#     V                               |
#  |____|____|____|____|____|.......|____|____| 
#  | 0  | 1  | 2  | 3  | 4  |.......|_Nx_|Nx+1| total Nx+2 entries  
#         |                                A
#         |                                |
#         | _______________________________|

# in the case of MPI, '0' and 'Nx+1' cells are threated as ghost cells.

    u   = np.zeros(Nx+2)
    u_1 = np.zeros(Nx+2)

    t1_list =[]
    t2_list =[]
    W0_list =[]
    alpha_list =[]

    for n in range(0, Nt):
        print(str(n))
        if model == 'EW': # Edward Wilkinson Model
            # Compute u at inner mesh points
            for i in range(1, Nx+1):
                eta =random.uniform(-0.5, 0.5)
                u[i] = u_1[i] + F*(u_1[i-1] - 2*u_1[i] + u_1[i+1]) + np.sqrt(12.0*dt) * eta

        if model == 'KPZ': # Kardar Parisi Zhang
            for i in range(1, Nx+1):
                eta =random.uniform(-0.5, 0.5)
                u[i] = u_1[i] + F * (u_1[i-1] - 2.0 * u_1[i] + u_1[i+1] + 0.125* (u_1[i+1] - u_1[i-1])**2.0 ) + np.sqrt(12.0*dt) * eta

        if model == 'BIO': # Biological model logistic r =0.15 k=1 ???
            for i in range(1, Nx+1):
                eta =random.uniform(-0.5, 0.5)
                u[i] = u_1[i] + F * (u_1[i-1] - 2.0 * u_1[i] + u_1[i+1] ) + dt * 0.15 * np.absolute(u_1[i]) * (1.0 - u_1[i]/10.) + np.sqrt(12.0*dt) * eta

        # Insert boundary conditions

        rank_l = (rank - 1) % size
        rank_r = (rank + 1) % size

#       print("left"  + str(rank_l))
#       print("right" + str(rank_r))
        if size > 1: # MPI message passing
            comm.Send([u[1], MPI.DOUBLE], dest=rank_l, tag=2*rank)
            comm.Send([u[Nx], MPI.DOUBLE], dest=rank_r, tag=2*rank+1)
            data = np.empty(1, dtype=np.float64)
            comm.Recv([data, MPI.DOUBLE], source=rank_l, tag=2*rank_l+1)
            u[0]=data
            data = np.empty(1, dtype=np.float64)
            comm.Recv([data, MPI.DOUBLE], source=rank_r, tag=2*rank_r)
            u[Nx+1]=data
        else: # serial case
             u[0] = u[Nx]
             u[Nx+1] = u[1]

        if (n % 5) == 0 :
            if size > 1:
               u_glo = np.asarray(comm.gather(u, root=0)).flatten() 
            else:
               u_glo = u
            if rank == 0:
                u_mean = u_glo[1:Nx+2].mean()
                w0     = np.sqrt( ( (u_glo[1:Nx+2]-u_mean)*(u_glo[1:Nx+2]-u_mean) ).sum()/Nx)
                t1_list.append(n*dt)
                W0_list.append(w0)
                t2_list.append(n*dt)
                file_out = 'pippo.npy'
                x_n=np.arange(Nx)

                alpha_list.append(CALC_ALPHA(x_n,u_glo,file_out))

        # Switch variables before next step
        u_1, u = u, u_1

    t1 = time.clock()

    return u_glo[1:Nx+2], t, t1-t0, t1_list, W0_list, t2_list, alpha_list

def main():
    # print command line arguments
    for i,arg in enumerate(sys.argv[1:]):
        if i == 0:
           model = arg
        if i == 1:
           rnd = arg
        if i >  1:
           raise TypeError(" max #arg = 2") 
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
    except:
        comm = 0
        size = 1
        rank = 0
    

    T  = 1000
#   T  = 10000

#   g  =  12.36 coupling constant 
#   nu = 0.5
#   sigma = 0.1
#   a  = np.sqrt(nu**3/sigma**2)
#   dx = 2.0*g*a/(5.0 np.sqrt(g)) = approx 5.0
    dx = 4.97
    dt = 0.65
    Ntot = 1000
    Nx = Ntot/size
    L    = dx*Nx
    Lglo = dx*Ntot
    F  = dt/dx**2
    x = np.linspace(0, Lglo, Ntot)   # mesh points in space

    if rnd == 'Det':
       print('Seed of random generator set to 19770213')
       random.seed(a=19770213)
        
    ut, t, tempo, t1_list, W_list, t2_list, alpha_list =  solver_FE_simple(model,comm,size,rank, dx, dt, Nx, L, Lglo, F, T)
    
    if rank == 0 :
	    
        time1 = np.asarray(t1_list)
        data2plot = np.asarray(W_list)

        ax1=plt.subplot(3, 1, 1)
        plt.plot(ut)
        plt.xlabel(r"$Space[\tilde{x}]$")
        plt.ylabel(r"$\tilde{u}(\tilde{x)}$")
        ax1.set_title(r'$\tilde{u}( \tilde{x} , \tilde{t} )$',fontsize=14)
        ax1.text(0.1,0.8,r'$t=$' '{:.0f}'.format(T),fontsize=12,transform=ax1.transAxes)

        ax2=plt.subplot(3, 1, 2)
        plt.loglog(time1,data2plot,linewidth=3,label=r'$W_{0}$',color='r')
        plt.xlabel(r"$Time[\tilde{t}]$")
        plt.ylabel(r"$\tilde{w}$")
        
        logt  = []
        logw0 = []
        
        for i,tt in enumerate(time1):
               if tt>0:
                   logt.append(np.log(tt))
                   logw0.append(np.log(data2plot[i]))
        try:
           from sklearn import linear_model, datasets
           ransac = linear_model.RANSACRegressor()
           vc=[]
# first fit
           x_ran=np.zeros((len(logt),1))
           y_ran=np.zeros((len(logw0),1))
           x_ran[:,0]  = logt
           y_ran[:,0] = logw0
           ransac.fit(x_ran, y_ran)
           inlier_mask = ransac.inlier_mask_
           outlier_mask = np.logical_not(inlier_mask)
        
           vc.append(ransac.estimator_.coef_[0][0])
           line_X = np.arange(x_ran.min(), x_ran.max(),0.1)[:, np.newaxis]
           line_y_ransac = ransac.predict(line_X)
           m0,b0 = np.polyfit(np.asarray(logt), np.asarray(logw0), 1)
           plt.title(r"$ \tilde{r} \approx \tilde{t}^\beta $",fontsize=14)
           ax2.text(0.1,0.75,r'$\beta=$' '{:01.2f}'.format(vc[0]),fontsize=12,transform=ax2.transAxes)
        except:
           print("An exception occurred check sklearn module or RANSAC procedure ... ")
           print(r"not estimating  beta ... ")
           plt.title(r"$ \tilde{w} \approx \tilde{t}^\beta$" ,fontsize=14)
        
        ax3=plt.subplot(3, 1, 3)
        time2 = np.asarray(t2_list)
        data2plot = np.asarray(alpha_list)
        plt.plot(time2,alpha_list)
        plt.xlabel(r"$Time [\tilde{t}]$")
        plt.ylabel(r"$\alpha$")
        plt.title(r"$ \tilde{w} \approx \tilde{x}^{\alpha}$",fontsize=14)
        
        plt.tight_layout()
        file_out = model + '.png'
        plt.savefig(file_out)
        

if __name__ == "__main__":
    main()

