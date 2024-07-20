from astropy import units as u
from astropy.coordinates import SkyCoord
import astropy.io.fits as fits
import astropy.wcs as wcs
import numpy as np



fov = 0.75 #deg

ra_c=15*(4+59.0/60.0+10.0/3600.0) #deg
dec_c=8+49.0/60.0 #deg


x=np.genfromtxt('ps_resolve_test.stat').T[8]
y=np.genfromtxt('ps_resolve_test.stat').T[9]
z=np.zeros(len(x))
w=np.zeros(len(x))

xy=np.array([[x],[y],[z],[w]])


w = wcs.WCS('abell_523_resolve_iteration_26.fits')

ra_list=[]
dec_list=[]


for i in range(0, len(x)):
    ra_list.append(w.all_pix2world([[x[i],y[i],0,0]], 1)[0][0])
    dec_list.append(w.all_pix2world([[x[i],y[i],0,0]], 1)[0][1])
    #print('ra', ra, 'deg', 'dec', dec,'deg')

ra=np.array(ra_list)
dec=np.array(dec_list)

diff_ra=ra_c-ra
diff_dec=dec-dec_c


print('ra_dii', diff_ra, 'deg', 'dec_diff', diff_dec,'deg')

f=open('differences_pix_wcs.txt', 'w')
for i in range (0,len(x)):  
    if abs(diff_ra[i]) < fov and abs(diff_dec[i])<fov:  
        f.write(str(diff_ra[i])+'deg$'+str(diff_dec[i])+'deg,')
f.close()

pass




