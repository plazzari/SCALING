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
from powernoise import powernoise
from mpi4py import MPI

def solver_FE_simple(model,comm,size,rank, dx, dt, Nx, L, Lglo, F, T):
    """
    Simplest expression of the computational algorithm
    using the Forward Euler method and explicit Python loops.
    For this method F <= 0.5 for stability.
    When solving ED or KPZ equation conditions are more stringent F ~0.05
    """
    N_MEMBERS = 100  # New parameter for ensemble size
    t0 = time.process_time()
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

    # Extend u and u_1 to include an additional dimension for N_MEMBERS
    if model == 'KOL': # Biological model logistic r =0.15 k=1 ???
        u   = np.zeros((Nx+2, N_MEMBERS)) 
        u_1 = np.zeros((Nx+2, N_MEMBERS))
    else:
        u   = np.ones((Nx+2, N_MEMBERS)) + np.random.normal(loc=0.0, scale = 0.33, size=(Nx+2, N_MEMBERS))
        u_1 = np.ones((Nx+2, N_MEMBERS)) + np.random.normal(loc=0.0, scale = 0.33, size=(Nx+2, N_MEMBERS))

    t1_list =[]
    t2_list =[]
    alpha_list =[]
    alpha_list_x =[]
    alpha_list_y =[]

    if model == "KOL":
       H = 3.5
       mean_CN = 0.5
       CN = 1.0;#powernoise(H, Nx)  # generate power law noise 
       CN -= np.amin(CN)  # set the lowest value to zero - all values >= 0
       CN *= mean_CN/np.mean(CN)  # set the mean to the desired value
       x_n=np.arange(Nx)
       file_out='snoopy'+str(Ntot)+'.np'
       CN_x=CALC_ALPHA(x_n,CN,file_out)[1]
       CN_y=CALC_ALPHA(x_n,CN,file_out)[2]
    else:
       CN_x=np.arange(Nx)
       CN_y=np.arange(Nx)

    W0=np.zeros((int(Nt/100)+1,N_MEMBERS))
    ALPHA_1=np.zeros((int(Nt/100)+1,N_MEMBERS))
    ALPHA_2=np.zeros((int(Nt/100)+1,N_MEMBERS))
    ALPHA_3=np.zeros((int(Nt/100)+1,N_MEMBERS))

    for n in range(0, Nt):
        print(str(n))
        if model == 'EW': # Edward Wilkinson Model
            # Compute u at inner mesh points
            for i in range(1, Nx+1):
                for m in range(N_MEMBERS):  # Loop over ensemble members
                    eta =random.uniform(-0.5, 0.5)
                    u[i, m] = u_1[i, m] + F*(u_1[i-1, m] - 2*u_1[i, m] + u_1[i+1, m]) + np.sqrt(12.0*dt) * eta

        if model == 'KPZ': # Kardar Parisi Zhang
            for i in range(1, Nx+1):
                for m in range(N_MEMBERS):  # Loop over ensemble members
                    eta =random.uniform(-0.5, 0.5)
                    u[i, m] = u_1[i, m] + F * (u_1[i-1, m] - 2.0 * u_1[i, m] + u_1[i+1, m] + 0.125* (u_1[i+1, m] - u_1[i-1, m])**2.0 ) + np.sqrt(12.0*dt) * eta

        if model == 'BIO': # Biological model logistic r =0.15 k=1 ???
            for i in range(1, Nx+1):
                for m in range(N_MEMBERS):  # Loop over ensemble members
                    eta =random.uniform(-0.5, 0.5)
                    u[i, m] = u_1[i, m] + F * (u_1[i-1, m] - 2.0 * u_1[i, m] + u_1[i+1, m] ) + dt * 0.15 * np.absolute(u_1[i, m]) * (1.0 - u_1[i, m]/10.) + np.sqrt(12.0*dt) * eta

        if model == 'BIO2': # Biological model logistic r =0.15 k=1 ???
            for i in range(1, Nx+1):
                for m in range(N_MEMBERS):  # Loop over ensemble members
                    eta =random.uniform(-0.5, 0.5)
                    u[i, m] = u_1[i, m] + F * (u_1[i-1, m] - 2.0 * u_1[i, m] + u_1[i+1, m] ) + dt * 0.000015 * np.absolute(u_1[i, m]) * ( 1.0 - u_1[i, m]/0.005) + np.sqrt(12.0*dt) * 0.00005 * eta * np.absolute(u_1[i, m])

        if model == 'KOL': # Biological model logistic r =0.15 k=1 ???
            for i in range(1, Nx+1):
                for m in range(N_MEMBERS):  # Loop over ensemble members
                    eta = max(CN[i-1],0) 
#               eta = max(CN[i-1] + np.random.normal(loc=0.0, scale = max(CN[i-1]*0.25, 10**(-10))),0) 
                    u[i, m] = max(0, u_1[i, m]) + F * (max(0, u_1[i-1, m]) - 2.0 * max(0, u_1[i, m]) + max(0, u_1[i+1, m]) ) + dt * (eta * max(0, u_1[i, m]) - u_1[i, m]**2/10.) 


        # Insert boundary conditions

        rank_l = (rank - 1) % size
        rank_r = (rank + 1) % size

#       print("left"  + str(rank_l))
#       print("right" + str(rank_r))
    
    
        if size > 1: # MPI message passing
            comm.Send([u[1,:], MPI.DOUBLE], dest=rank_l, tag=2*rank)
            comm.Send([u[Nx,:], MPI.DOUBLE], dest=rank_r, tag=2*rank+1)
            data = np.empty([1,N_MEMBERS], dtype=np.float64)
            comm.Recv([data, MPI.DOUBLE], source=rank_l, tag=2*rank_l+1)
            u[0,:]=data
            data = np.empty([1,N_MEMBERS], dtype=np.float64)
            comm.Recv([data, MPI.DOUBLE], source=rank_r, tag=2*rank_r)
            u[Nx+1,:]=data
        else: # serial case
             u[0,:] = u[Nx,:]
             u[Nx+1,:] = u[1,:]

        if (n % 100) == 0:
            if size > 1:
               aux = comm.gather(u[1:Nx+1,:], root=0)
               # Gather the data from all processes
               if rank == 0:    
                   Nxtot   = size * Nx  # Total number of computational cells
                   # Reshape the gathered data into a 2D array
                   aux = np.asarray(aux)  # Convert to numpy array
                   u_glo = np.reshape(aux, (Nxtot, N_MEMBERS)) 

            else:
               u_glo = u

            if rank == 0:
                t1_list.append(n*dt)
                t2_list.append(n*dt)             
                for m in range(N_MEMBERS):
                    u_mean = u_glo[1:Nxtot, m].mean()
                    W0[int(n/100), m] = np.sqrt(((u_glo[1:Nxtot, m] - u_mean) ** 2).sum() / Nxtot)

                    file_out = 'stat_'+str(Nxtot)+'.npy'
                    x_n=np.arange(Nxtot)

                    ALPHA_1[int(n/100),m]=CALC_ALPHA(x_n,u_glo[:,m],file_out)[0]
#                    ALPHA_2[int(n/100),m]=CALC_ALPHA(x_n,u_glo[:,m],file_out)[1]
#                    ALPHA_3[int(n/100),m]=CALC_ALPHA(x_n,u_glo[:,m],file_out)[2]
                   
            

#               alpha_list.append(CALC_ALPHA(x_n[0:100]  ,   u_glo[0:100],file_out)[0])
#               alpha_list_x.append(CALC_ALPHA(x_n[0:100],   u_glo[0:100],file_out)[1])
#               alpha_list_y.append(CALC_ALPHA(x_n[0:100],   u_glo[0:100],file_out)[2])

        # Switch variables before next step
        u_1, u = u, u_1

    t1 = time.process_time()

    return u_glo[1:Nxtot,:], t, t1-t0, t1_list, W0, t2_list, ALPHA_1,ALPHA_2,ALPHA_3,CN_x,CN_y

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
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
    except:
        comm = 0
        size = 1
        rank = 0
    

    T  = 10000

#   g  =  12.36 coupling constant 
#   nu = 0.5
#   sigma = 0.1
#   a  = np.sqrt(nu**3/sigma**2)
#   dx = 2.0*g*a/(5.0 np.sqrt(g)) = approx 5.0
    dx = 4.97
    dt = 0.65
    Ntot = 250
    Nx = int(Ntot/size)
    L    = dx*Nx
    Lglo = dx*Ntot
    F  = dt/dx**2
    x = np.linspace(0, Lglo, Ntot)   # mesh points in space

    if rnd == 'Det':
       print('Seed of random generator set to 19770213')
       random.seed(a=19770213)
        
    ut, t, tempo, t1_list, W, t2_list, alpha, alpha_x, alpha_y, CN_x, CN_y =  solver_FE_simple(model,comm,size,rank, dx, dt, Nx, L, Lglo, F, T)
    
    if rank == 0 :
	    
        time1 = np.asarray(t1_list)
        data2plot = np.mean(W,axis=1)   

        ax1=plt.subplot(2, 2, 1)
        plt.plot(ut)
        plt.xlabel(r"$Space[\tilde{x}]$")
        plt.ylabel(r"$\tilde{u}(\tilde{x)}$")
        ax1.set_title(r'$\tilde{u}( \tilde{x} , \tilde{t} )$',fontsize=14)
        ax1.text(0.1,0.8,r'$t=$' '{:.0f}'.format(T),fontsize=12,transform=ax1.transAxes)

#        ax2=plt.subplot(2, 2, 2)
#        plt.loglog(np.mean(alpha_x,axis=1),np.mean(alpha_y,axis=1),linewidth=3,label=r'$W_{0}$',color='r')
        
        #plt.loglog(np.mean[]alpha_list_x[-1],alpha_list_y[-1])
#        plt.xlabel(r"$Space[\tilde{x}]$")
#        plt.ylabel(r"$\tilde{u}(\tilde{x)}$")
#        ax1.set_title(r'$\tilde{u}( \tilde{x} , \tilde{t} )$',fontsize=14)

        ax3=plt.subplot(2, 2, 3)
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
           plt.title(r"$ \tilde{W} \approx \tilde{t}^\beta $",fontsize=14)
           ax3.text(0.1,0.75,r'$\beta=$' '{:01.2f}'.format(vc[0]),fontsize=12,transform=ax3.transAxes)
        except:
           print("An exception occurred check sklearn module or RANSAC procedure ... ")
           print(r"not estimating  beta ... ")
           plt.title(r"$ \tilde{w} \approx \tilde{t}^\beta$" ,fontsize=14)
        

        ax4=plt.subplot(2, 2, 4)
        time2 = np.asarray(t2_list)
        data2plot = np.mean(alpha,axis=1)
        plt.plot(time2,data2plot,linewidth=3,label=r'$W_{0}$',color='r')

        plt.xlabel(r"$Time [\tilde{t}]$")
        plt.ylabel(r"$\alpha$")
        plt.title(r"$\tilde{w} \approx \tilde{x}^{\alpha}$",fontsize=14)
        
        plt.tight_layout()
        file_out = model + '_' + str(Ntot) + '.png'
        plt.savefig(file_out)
        

if __name__ == "__main__":
    main()

