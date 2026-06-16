import pickle
import numpy as np
import matplotlib.pyplot as plt

L_list=[500,250,125]
MODEL='KPZ' # options are EW or KPZ

plt.figure()  # Creazione di una nuova figura
if MODEL == 'KPZ':
    beta=1./3.
    alpha=1./2.
    z=alpha/beta
    reference_slope = beta  
    label_ref=f'Reference slope 1/3'
if MODEL == 'EW':
    beta=1./4.
    alpha=1./2.
    z=alpha/beta
    reference_slope = beta  
    label_ref=f'Reference slope 1/4'

for i,L in enumerate(L_list):
    data_file=MODEL+'_'+str(L)+'.pkl'
    scT=L**z
    scW=L**alpha
    with open(data_file, 'rb') as f:
        data=pickle.load(f)
    plt.plot(np.log( data[0][1:]/scT ), np.log( data[1][1:]/scW) , label=f'L={L}')
    if i == 0:
        x_ref = np.log( data[0][1:]/scT )  # Range di valori x per la curva di riferimento
        c=-reference_slope*np.log( data[0][1]/scT )+np.log( data[1][1]/scW)

# Definizione della pendenza e della curva di riferimento
y_ref = reference_slope * x_ref + c  # Calcolo dei valori y corrispondenti
plt.plot(x_ref, y_ref, '--', label=label_ref)  # Aggiunta della curva di riferimento

plt.xlabel(r'Scaled Time $\log(T/L^{z})$',fontsize=18)  
plt.ylabel(r'Scaled W $\log(W/L^{\alpha})$',fontsize=18) 

# Rimozione dei numeri sui tick degli assi
plt.xticks([], [])
plt.yticks([], [])

plt.legend() 

plot_file='Scaling_'+MODEL+'.png'
plt.savefig(plot_file)
plt.show()
