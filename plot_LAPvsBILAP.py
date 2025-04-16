import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

def main():
    # print command line arguments
    for i,arg in enumerate(sys.argv[1:]):
        if i == 0:
           wn = arg
        if i >  0:
           raise TypeError(" max #arg = 1") 

    L_list=[250]
    MODEL=['LAP','BILAP'] # options are LAP or BILAP

    fig, axes = plt.subplots(8, 1, figsize=(8, 8 * 4))  # Create subplots

    decay=np.zeros((2,8))
    L=L_list[0]
    for i,mod in enumerate(MODEL):
        for ax, timeframe in zip(axes, range(8)):
            data_file = 'LAPvsBILAP_data/' + mod + '_' + str(L) +'_' + str(wn) + '.pkl'
            with open(data_file, 'rb') as f:
                data = pickle.load(f)
            ax.plot(data[1][:,timeframe], label=f'{mod}')
            decay[i][timeframe]=np.mean(data[1][int(L/2),timeframe])
            # Add a vertical line at the midpoint of the x-axis
            ax.axvline(x=L/2, color='red', linestyle='--')
        ax.legend()

    plot_file = 'LAPvsBILAP_plots/LAPvsBILAP' + str(L) +'_' + str(wn) + '.png'
    plt.savefig(plot_file)


    # Add a second figure to visualize decay
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    for i, mod in enumerate(MODEL):
        ax2.plot(range(8), decay[i,:], marker='o', label=f'{mod}')
    tauLap = np.log10(decay[0,0]/decay[0,-1])/8
    tauBILap = np.log10(decay[1,0]/decay[1,-1])/8
    ax2.set_title('Decay Rate ratio ' + str(tauLap/tauBILap))
    ax2.set_xlabel('Timeframe')
    ax2.set_ylabel('Decay')
    ax2.legend()

    decay_plot_file = 'LAPvsBILAP_plots/LAPvsBILAP_decay' + str(L) +'_' + str(wn) + '.png'
    plt.savefig(decay_plot_file)
    


if __name__ == "__main__":
    main()
