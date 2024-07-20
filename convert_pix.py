from astropy import units as u
from astropy.coordinates import SkyCoord
import astropy.io.fits as fits
import astropy.wcs as wcs
import numpy as np

half_fov = 0.5 #deg

cell= 1.0/512.0 #deg/pix

#x=np.genfromtxt('point_sources_resolve.stat').T[8]
#y=np.genfromtxt('point_sources_resolve.stat').T[9]
x=np.genfromtxt('ps_resolve_test.stat').T[8]
y=np.genfromtxt('ps_resolve_test.stat').T[9]

xc = 255
yc = 255 

diff_x = (x-xc)*cell
diff_y = (y-yc)*cell 


f=open('differences_pix.txt', 'w')
for i in range (0,len(x)):  
    if diff_x[i] < half_fov and diff_y[i] < half_fov:  
        f.write(str(diff_x[i])+'deg$'+str(diff_y[i])+'deg,')
    else:
        print(str(x[i])+','+str(y[i])+','+str(diff_x[i])+'deg$'+str(diff_y[i])+'deg,')
f.close()

pass