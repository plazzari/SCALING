import numpy as np
import pylab as pl
#from tempfile import TemporaryFile
 
def CALC_ALPHA(x,h,file_out):
    Lx=len(x)
    Ly=len(h)
    mydata=[]
    for ccc in range(Lx):
       mydata.append((x[ccc],h[ccc]))

    mydata=np.asarray(mydata)
 
    # computing the fractal dimension
    #considering only scales in a logarithmic list
    # looping over several scales
    np.save(file_out, mydata)
    minx=np.min(x)
    maxx=np.max(x)
    minh=np.min(h)
    maxh=np.max(h)

    
#   scales=np.logspace(1, 8, num=20, endpoint=False, base=2)
    scales=np.logspace(1, 12, num=24, endpoint=False, base=2)
    Ns=[]
    Ws=[]
    scales_fit=[]
    for scale in scales:
    # computing the histogram
#        print " ======= Scale ", scale
         myedges_x= np.arange(minx,maxx,scale)

         Nint=len(myedges_x)

         W=np.zeros( Nint )

         if (len(myedges_x) > 1): 

             for i in range(Nint):

                 left  = int(myedges_x[i])

                 if i < Nint-1:

                   right = int(myedges_x[i+1])

                 else:

                   right = maxx

                 mydatax=mydata[:,0]

#                print " ======= Left ", left,  " ======= Right ", right

                 if right-left >1:
                     W[i]  = np.nanstd(mydata[ (mydatax >= left) &  (mydatax < right)    ,1])
                 else:
                     W[i]  = np.nan

             Ws.append(np.nanmean(W))

             scales_fit.append(scale)

#        print " ======= Scale ", Lx,Ly,scale, np.nanmean(W)
    
    try:
        coeffs=np.polyfit(np.log(scales_fit), np.log(Ws), 1)
    except:
        coeffs=[np.nan,np.nan]

    return coeffs[0] 
 
