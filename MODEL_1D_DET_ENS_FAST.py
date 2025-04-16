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
import pickle

def solver_FE_simple(model,wn,comm,size,rank, dx, dt, Nx, L, Lglo, F, Flap, T):
    """
    Simplest expression of the computational algorithm
    using the Forward Euler method and explicit Python loops.
    For this method F <= 0.5 for stability.
    When solving ED or KPZ equation conditions are more stringent F ~0.05
    """
    N_MEMBERS = 1  # New parameter for ensemble size
    t0 = time.process_time()
    print("dt =" + str(dt))
    Nt = int(round(T/float(dt)))
    t = np.linspace(0, T, Nt+1)   # mesh points in time

# We add two cells 0 and N+1 position for periodic boundary conditions 
# and GHOST cells in the case of MPI 
#
#  |____|____|____|____|____|.......|____|____| 
#  | 0  | 1  | 2  | 3  | 4  |.......| Nx |Nx+1| total Nx+2 entries  
#  |xxxx|______Nx computational cells____|xxxx| 
#
#     At each time step periodic Boundary Conditions are applied as follows
#      _______________________________
#     |                               |
#     |                               |
#     V                               |
#  |____|____|____|____|____|.......|____|____| 
#  | 0  | 1  | 2  | 3  | 4  |.......| Nx |Nx+1| total Nx+2 entries  
#         |                                A
#         |                                |
#         | _______________________________|

# in the case of MPI, '0' and 'Nx+1' cells are threated as ghost cells.

# We add four cells 0,1 and N,N+1 position for periodic boundary conditions
# and GHOST cells in the case of MPI
#
#  |____|____|____|____|____|........|____|____|____|____|____|
#  | 0  | 1  | 2  | 3  | 4  |........|Nx-3|Nx-2|Nx-1| Nx |Nx+1| total Nx+2 entries
#  |xxxx|xxxx|____Nx-2 computational cells___________|xxxx|XXXX|
#
#     At each time step periodic Boundary Conditions are applied as follows
#      _____________________________________    
#     |                                    |      
#     |    ________________________________|____
#     |   |                                |    |   
#     V   V                                |    | 
#  |____|____|____|____|____|.......|____|____|____|____|____|
#  | 0  | 1  | 2  | 3  | 4  |.......|Nx-3|Nx-2|Nx-1| Nx |Nx+1| total Nx+2 entries
#              |    |                                 A    A
#              |    |_________________________________|____|
#              |                                      |
#              |______________________________________|

# in the case of MPI, '0' and 'Nx+1' cells are threated as ghost cells.

    # Extend u and u_1 to include an additional dimension for N_MEMBERS
    u   = np.zeros((Nx+2, N_MEMBERS)) 
    u_1 = np.zeros((Nx+2, N_MEMBERS))
    u_glo= np.zeros((Nx+2, int(Nt/100)+1)) 
    myx = np.arange(Nx+2) 
    ave=Nx/2
    sigma=1. # gives negative values
    sigma=10. # gives positive values
    # wn wave number
    
    k=np.pi/(wn*dx)
    phase= np.pi/2*(1-Nx/wn)

    gauss = np.exp(-(myx-ave)**2./sigma**2) 
    u_1=gauss.reshape(*gauss.shape, 1) # assume N_MENBERS=1
    wave = np.sin(k*myx*dx+phase) 
    u_1=wave.reshape(*wave.shape, 1) # assume N_MENBERS=1
    u=u_1.copy() # copy the initial condition to u

    t1_list =[]
    count=0
    for n in range(0, Nt):
        print(str(n))

        if model == 'LAP': # Laplacian Diffusion
            # Compute u at inner mesh points
            for i in range(1, Nx+1):
                for m in range(N_MEMBERS):  # Loop over ensemble members
                    u[i, m] = u_1[i, m] + F*(u_1[i-1, m] - 2*u_1[i, m] + u_1[i+1, m]) 

        if model == 'BILAP': # BILaplacian Diffusion
            # Compute u at inner mesh points
            # https://en.wikipedia.org/wiki/Finite_difference_coefficient
            for i in range(2, Nx):
                for m in range(N_MEMBERS):  # Loop over ensemble members
                    u[i, m] = u_1[i, m] - Flap*(u_1[i-2, m] -4*u_1[i-1, m] + 6*u_1[i, m] -4*u_1[i+1, m] + u_1[i+2, m]) 

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
             if model == 'BILAP': # BILaplacian Diffusion
                 u[0,:] = u[Nx-2,:]
                 u[1,:] = u[Nx-1,:]
                 u[Nx,:] = u[2,:]
                 u[Nx+1,:] = u[3,:]
             else:
                 u[0,:] = u[Nx,:]
                 u[Nx+1,:] = u[1,:]

        if (n % 100) == 0:
            Nxtot   = size * Nx # Total number of computational cells
            if size > 1:
                aux = comm.gather(u[2:Nx,:], root=0)
                # Gather the data from all processes
                if rank == 0:    
                   # Reshape the gathered data into a 2D array
                   aux = np.asarray(aux)  # Convert to numpy array
                   u_glo = np.reshape(aux, (Nxtot, N_MEMBERS))
                else:
                    u_glo = np.empty((Nxtot, N_MEMBERS), dtype=np.float64)
                comm.Bcast(u_glo, root=0)
            else:
                u_glo[:,count] = u[:,0]
                count += 1

            if rank == 0:
                t1_list.append(n*dt)
            
            x_n=np.arange(Nxtot)

        # Switch variables before next step
        u_1, u = u, u_1

    t1 = time.process_time()

    return u_glo[0:Nxtot,:],  t1_list

def main():
    # print command line arguments
    for i,arg in enumerate(sys.argv[1:]):
        if i == 0:
           model = arg
        if i == 1:
           wn = int(arg)
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
    

    T  = 500

#   g  =  12.36 coupling constant 
#   nu = 0.5
#   sigma = 0.1
#   a  = np.sqrt(nu**3/sigma**2)
#   dx = 2.0*g*a/(5.0 np.sqrt(g)) = approx 5.0
    dx = 4.97
    dt = 0.65
    Ntot = 250 # altri test 500, 250, 125
    Nx = int(Ntot/size)
    L    = dx*Nx
    Lglo = dx*Ntot
    F  = dt/dx**2
    #Flap = dt/dx**4.
    Flap = F*(dx**2)/8.
    x = np.linspace(0, Lglo, Ntot)   # mesh points in space

        
    ut, t1_list =  solver_FE_simple(model,wn,comm,size,rank, dx, dt, Nx, L, Lglo, F, Flap, T)
    

    # Only rank 0 process will plot the results
    if rank == 0 :

	    
        time1 = np.asarray(t1_list)
        data2plot = ut

        file_out = 'LAPvsBILAP_data/' + model + '_' + str(Ntot) + '_'+ str(wn) +'.pkl'
        saveObject = (time1, data2plot)
        with open(file_out, 'wb') as f: 
            pickle.dump(saveObject, f)
   
        


if __name__ == "__main__":
    main()

