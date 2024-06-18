# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 13:06:46 2023

@author: yansh00
"""
from scipy.io import netcdf

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
 

##$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#import matplotlib.pyplot as plt
#import numpy as np
from scipy.io import netcdf
import matplotlib
import math
import os

##########################################################################################
# Ground_True	Guassian = 10, floating32, Range = [0..1]
##生产ground truth的打点图。
# array_2d = np.zeros((360, 360), dtype=float)    # Create an array of zeros with the shape 360x360
# array_2d[180, 180] = 1                          # Place a single point with value 1 in the center of the array
# array_2d = gaussian_filter(array_2d, sigma=10)  # Apply Gaussian filter with sigma = 10
# array_2d /= array_2d.max()                      # Normalize the array to have values from 0 to 1
# plt.imshow(array_2d)                            #plt.savefig('./image_png/img_' + str(t) + '.png')
# plt.show()
def dotImage(array_2d, x, y): 
    array_2d_empty = np.zeros(np.shape(array_2d), dtype=float)
    array_2d_empty[x, y] = 1    
    array_2d_empty = gaussian_filter(array_2d_empty, sigma=10)  # Apply Gaussian filter with sigma = 10
    array_2d_empty /= array_2d_empty.max() 
    return array_2d_empty + array_2d


def Lat2Cube(theta, phi): 
    xyz = np.zeros(3)
    xyz[0] = np.sin(np.pi*(90-theta)/180.0) * np.cos(np.pi*phi/180.0)
    xyz[1] = np.sin(np.pi*(90-theta)/180.0) * np.sin(np.pi*phi/180.0)
    xyz[2] = np.cos(np.pi*(90-theta)/180.0)
    
    xyz = xyz/np.reshape(np.repeat(np.amax(abs(xyz),axis=1),3),3)
    return xyz

def latlon2cubesphere(theta, phi): 
    N = len(theta)
    xyz = np.zeros([N,3])
    xyz[:,0] = np.sin(np.pi*(90-theta)/180.0) * np.cos(np.pi*phi/180.0)
    xyz[:,1] = np.sin(np.pi*(90-theta)/180.0) * np.sin(np.pi*phi/180.0)
    xyz[:,2] = np.cos(np.pi*(90-theta)/180.0)
    
    xyz = xyz/np.reshape(np.repeat(np.amax(abs(xyz),axis=1),3),[N,3])
    
    
    return xyz

def cubesphere2t(xyz):
    N = xyz.shape[0]

    xyz = np.round(xyz,decimals=int(np.floor(-np.log10(1/N))))
    
    # Find top map
    c_ind = np.argwhere(xyz[:,2]>1-1/N)
    x_cord = np.unique(xyz[c_ind[:,0],0])
    y_cord = np.unique(xyz[c_ind[:,0],1])
        
    ind_top = np.zeros([len(x_cord),len(y_cord)],dtype=int)
    
    for jx in range(len(x_cord)):
        cs_ind = np.argwhere(xyz[c_ind[:,0],0]==x_cord[jx])
        css_ind = np.argsort(xyz[c_ind[cs_ind[:,0],0],1])
        ind_top[jx,:] = c_ind[cs_ind[css_ind,0],0]
            
    # Find bottom map
    
    c_ind = np.argwhere(xyz[:,2]<-1+1/N)
    x_cord = np.unique(xyz[c_ind[:,0],0])
    y_cord = np.unique(xyz[c_ind[:,0],1])
        
    ind_bottom = np.zeros([len(x_cord),len(y_cord)],dtype=int)
    
    for jx in range(len(x_cord)):
        cs_ind = np.argwhere(xyz[c_ind[:,0],0]==x_cord[jx])
        css_ind = np.argsort(xyz[c_ind[cs_ind[:,0],0],1])
        ind_bottom[jx,:] = c_ind[cs_ind[css_ind,0],0]
    

    # Find side map
    c_ind = np.argwhere((abs(xyz[:,2])<1-1/N)&(xyz[:,0]<-1+1/N))
    y_cord = np.unique(xyz[c_ind[:,0],1])
    z_cord = np.unique(xyz[c_ind[:,0],2])
    
    ind_sides = np.zeros([len(y_cord),len(z_cord)],dtype=int)
    
    for jz in range(len(z_cord)):
        cs_ind = np.argwhere(xyz[c_ind[:,0],2]==z_cord[jz])
        css_ind = np.argsort(xyz[c_ind[cs_ind[:,0],0],1])
        ind_sides[:,jz] = c_ind[cs_ind[css_ind,0],0]
        
    c_ind = np.argwhere((abs(xyz[:,2])<1-1/N)&(xyz[:,1]>1-1/N))
    x_cord = np.unique(xyz[c_ind[:,0],0])
    z_cord = np.unique(xyz[c_ind[:,0],2])
    
    ind_sides_c = np.zeros([len(x_cord),len(z_cord)],dtype=int)
    
    for jz in range(len(z_cord)):
        cs_ind = np.argwhere(xyz[c_ind[:,0],2]==z_cord[jz])
        css_ind = np.argsort(xyz[c_ind[cs_ind[:,0],0],0])
        ind_sides_c[:,jz] = c_ind[cs_ind[css_ind,0],0]
    
    
    ind_sides = np.append(ind_sides,ind_sides_c,axis=0)
    
    c_ind = np.argwhere((abs(xyz[:,2])<1-1/N)&(xyz[:,0]>1-1/N))
    y_cord = np.unique(xyz[c_ind[:,0],1])
    z_cord = np.unique(xyz[c_ind[:,0],2])
    
    ind_sides_c = np.zeros([len(y_cord),len(z_cord)],dtype=int)
    
    for jz in range(len(z_cord)):
        cs_ind = np.argwhere(xyz[c_ind[:,0],2]==z_cord[jz])
        css_ind = np.argsort(-xyz[c_ind[cs_ind[:,0],0],1])
        ind_sides_c[:,jz] = c_ind[cs_ind[css_ind,0],0]
    
    ind_sides = np.append(ind_sides,ind_sides_c,axis=0)
    
    c_ind = np.argwhere((abs(xyz[:,2])<1-1/N)&(xyz[:,1]<-1+1/N))
    x_cord = np.unique(xyz[c_ind[:,0],0])
    z_cord = np.unique(xyz[c_ind[:,0],2])
    
    ind_sides_c = np.zeros([len(x_cord),len(z_cord)],dtype=int)
    
    for jz in range(len(z_cord)):
        cs_ind = np.argwhere(xyz[c_ind[:,0],2]==z_cord[jz])
        css_ind = np.argsort(-xyz[c_ind[cs_ind[:,0],0],0])
        ind_sides_c[:,jz] = c_ind[cs_ind[css_ind,0],0]
    
    ind_sides = np.append(ind_sides,ind_sides_c,axis=0)
    
    return ind_sides, ind_top, ind_bottom
##$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$



#with open("cyclones_stitch_hires_5yrs.original.txt") as f:
with open("TC_tracks_1mo.txt") as f:
    lines = f.readlines()
    
num_storms = 0
max_len = 0
for i in range(0, np.size(lines)):
    if lines[i][0] == "s":
        num_storms = num_storms + 1
        num_points = 0
    else:
        num_points = num_points + 1
        max_len = max(max_len, num_points)
print('number of original storms', num_storms)
print('max length of original storms', max_len)



latmc_1d0  = np.zeros((max_len*num_storms, 1))
longmc_1d0 = np.zeros((max_len*num_storms, 1))
#
idxmc = np.zeros((max_len, num_storms), dtype='int')
latmc = np.zeros((max_len, num_storms))
longmc = np.zeros((max_len, num_storms))
vsmc = np.zeros((max_len, num_storms))
pcmc = np.zeros((max_len, num_storms))
monthmc = np.zeros((max_len, num_storms),dtype='int')
yearmc = np.zeros((max_len, num_storms),dtype='int')
daymc = np.zeros((max_len, num_storms),dtype='int')
hourmc = np.zeros((max_len, num_storms), dtype='int')
index = 0
year_start = int(lines[0].split("\t")[2])
year_end = year_start


plt.figure(figsize=(20,10))
#@@ax = plt.axes(projection=ccrs.PlateCarree())

#plt.title('Australia')
#@@ax.set_extent([-180, 180, -45, 45], ccrs.PlateCarree())
#@@ax.coastlines(resolution='110m')

#plt.show()
#quit()

cnt = 0
for i in range(0, np.size(lines)):
    if lines[i][0] == "s":   #start	21	56	1	7	6
        index = index + 1
        year = int(lines[i].split("\t")[2])
        year_end = max(year, year_start)
        k = 0
    else:
        cnt = cnt + 1
        k   = k + 1
        line_split = lines[i].split("\t")
        idxmc[k - 1, index - 1] = float(lines[i].split("\t")[1]) 
        longmc[k - 1, index - 1] = float(lines[i].split("\t")[2])
        latmc[k - 1, index - 1] = float(lines[i].split("\t")[3])
        longmc_1d0[cnt-1] = longmc[k - 1, index - 1]
        latmc_1d0[cnt-1]  = latmc[k - 1, index - 1]
        
        #how to sort out "ind_sides" cyclones??
        #index is No. of cyclone tracks
        #k is the footprint of cyclone track. (footprint = days of cyclone*stamps_in_a_day)
      
# 1st month, 31 days
        pcmc[k - 1, index - 1]    = float(lines[i].split("\t")[4])
        # Convert wind speed from units m/s to knot by multiplying 1.94
        vsmc[k - 1, index - 1]    = float(lines[i].split("\t")[5]) * 1.94
        yearmc[k - 1, index - 1]  = float(lines[i].split("\t")[6])
        monthmc[k - 1, index - 1] = float(lines[i].split("\t")[7])
#@@
        daymc[k - 1, index - 1]  = float(lines[i].split("\t")[8])  #30 days
        hourmc[k - 1, index - 1] = float(lines[i].split("\t")[9])
#@@            
        marksize_wind = vsmc[k - 1, index - 1]/10
#latmc_1d = np.empty(cnt, dtype=float)
#longmc_1d = np.empty(cnt, dtype=float) 


# for i in range(cnt):
#     latmc_1d = np.insert(latmc_1d, i, latmc_1d0[i])
#     longmc_1d = np.insert(longmc_1d, i, longmc_1d0[i])
# xyz = latlon2cubesphere(latmc_1d, longmc_1d)   
# ind_sides, ind_top, ind_bottom = cubesphere2t(xyz)
      
########################################################################################
# Load lat lon file and cube sphere mapping 
cs_file = netcdf.NetCDFFile('ne120np4_latlon.100310.nc','r')  #reference file, variable to coordination. 1D 100K/time_stamp
lat = np.copy(cs_file.variables['lat'][:])
lon = np.copy(cs_file.variables['lon'][:])
cs_file.close()

xyz = latlon2cubesphere(lat, lon)   #Cooridination from sphere to 2D
ind_sides, ind_top0, ind_bottom0 = cubesphere2t(xyz) #ind_sides(1444,359), ind_top(361,361), ind_bottom(361,361)
#ind_sides_1=ind_sides[:360,:], ind_sides_2 = ind_sides[720:720+360,:], etc 这样把ind_side分成四个
ind_sides_1 = np.zeros([360,360])
ind_sides_2 = np.zeros([360,360])
ind_sides_3 = np.zeros([360,360])
ind_sides_4 = np.zeros([360,360])
ind_sides_1[:360,:359] = ind_sides[    : 360,:]
ind_sides_2[:360,:359] = ind_sides[ 361: 361 +360,:]
ind_sides_3[:360,:359] = ind_sides[ 722: 722 +360,:]
ind_sides_4[:360,:359] = ind_sides[1083:1083 +360,:]
ind_top     = ind_top0   [:360, :360]
ind_bottom  = ind_bottom0[:360, :360]

print(ind_sides_1.shape, ind_sides_2.shape, ind_top.shape)

#1956, 1960:  [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
#1957, 58, 59 [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
month_dict28 = np.array([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
month_dict28 = np.cumsum(month_dict28)
month_dict29 = np.array([0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
month_dict29 = np.cumsum(month_dict29)
month_dict   = month_dict28

# Data_Dimenision defined before (max_len, num_storms)
table_mark = np.zeros([max_len, num_storms], dtype='int')
table_rc   = np.zeros([max_len, num_storms, 2], dtype='int')
########################################################################################
#@@ 把球面转化成6个2D平面后，在2D平面上根据cyclone的位置，生存Gaussian heatmap
ground_true = np.zeros([360,360])
total_day = 366+365+365+365+366
#total_img = total_day*4*6
total_img = 150*4*6
print('Total days = ', str(total_day), ', points =' + str(total_day*4) + ', images = ', str(total_img))
#for s in range(total_img):
#    # Initial all the ground true image
#    np.save('./Image_2D_true/truth_' + str(s) + '.npy', ground_true)
#    plt.savefig('./Image_2D_true/truth_' + str(s) + '.png')


n_cyc = 0
for s in range(20): #num_storms):  #4):
    steps = np.count_nonzero(daymc[:,s])  #Find days 
    for l in range(steps):
        ### Get global_ID from Storm_2D_Array(max_len, num_storms) in coordinate, y/m/d/h
        #  (longmc, latmc, yearmc, monthmc, daymc, hourmc)
        if yearmc[l,s]==56:   
            days = 0
            month_dict   = month_dict29
        elif yearmc[l,s]==57:   
            days = 366
            month_dict   = month_dict28
        elif yearmc[l,s]==58:   
            days = 366 + 365
            month_dict   = month_dict28
        elif yearmc[l,s]==59:   
            days = 366 + 365 + 365
            month_dict   = month_dict28
        elif yearmc[l,s]==60:   
            days = 366 + 365 + 365 + 365
            month_dict   = month_dict29
        else: 
            print('Year out of range')
            
        idx = days*4 + month_dict[monthmc[l,s]-1]*4 + (daymc[l,s]-1)*4 + hourmc[l,s]/6
        time = int( idx ) # % 120)
#        print(time)
#        print(idxmc[l,s])
        n_cyc +=1
        if  (np.where(ind_sides_1==idxmc[l,s])[0].shape[0] > 0):
            [r, c] = np.where(ind_sides_1==idxmc[l,s])
            table_mark[l,s] = 1
            table_rc[l,s,0] = r
            table_rc[l,s,1] = c
            gID = time*6 + 1
            print('side 1:' + str(r) + str(c)+ ' timeID= '+str(time)+', gID= '+str(gID)+',  Y/M/D/H='+str(yearmc[l,s])+','+str(monthmc[l,s])+','+str(daymc[l,s])+','+str(hourmc[l,s]) ) 
        elif (np.where(ind_sides_2==idxmc[l,s])[0].shape[0] > 0):
            [r, c] = np.where(ind_sides_2==idxmc[l,s])
            table_mark[l,s] = 2
            table_rc[l,s,0] = r
            table_rc[l,s,1] = c
            gID = time*6 + 2
            print('side 2:' + str(r) + str(c)+ ' timeID= '+str(time)+', gID= '+str(gID)+',  Y/M/D/H='+str(yearmc[l,s])+','+str(monthmc[l,s])+','+str(daymc[l,s])+','+str(hourmc[l,s]) ) 
        elif (np.where(ind_sides_3==idxmc[l,s])[0].shape[0] > 0):
            [r, c] = np.where(ind_sides_3==idxmc[l,s])
            table_mark[l,s] = 3
            table_rc[l,s,0] = r
            table_rc[l,s,1] = c
            gID = time*6 + 3
            print('side 3:' + str(r) + str(c)+ ' timeID= '+str(time)+', gID= '+str(gID)+',  Y/M/D/H='+str(yearmc[l,s])+','+str(monthmc[l,s])+','+str(daymc[l,s])+','+str(hourmc[l,s]) ) 
        elif (np.where(ind_sides_4==idxmc[l,s])[0].shape[0] > 0):
            [r, c] = np.where(ind_sides_4==idxmc[l,s])
            table_mark[l,s] = 4
            table_rc[l,s,0] = r
            table_rc[l,s,1] = c
            gID = time*6 + 4
            print('side 4:' + str(r) + str(c)+ ' timeID= '+str(time)+', gID= '+str(gID)+',  Y/M/D/H='+str(yearmc[l,s])+','+str(monthmc[l,s])+','+str(daymc[l,s])+','+str(hourmc[l,s]) ) 
        elif (np.where(ind_top==idxmc[l,s])[0].shape[0] > 0):
            [r, c] = np.where(ind_top==idxmc[l,s])
            table_mark[l,s] = 5
            table_rc[l,s,0] = r
            table_rc[l,s,1] = c
            gID = time*6 + 5
            print('top:   ' + str(r) + str(c)+ ' timeID= '+str(time)+', gID= '+str(gID)+',  Y/M/D/H='+str(yearmc[l,s])+','+str(monthmc[l,s])+','+str(daymc[l,s])+','+str(hourmc[l,s]) ) 
        elif (np.where(ind_bottom==idxmc[l,s])[0].shape[0] > 0):
            [r, c] = np.where(ind_bottom==idxmc[l,s])
            table_mark[l,s] = 6
            table_rc[l,s,0] = r
            table_rc[l,s,1] = c
            gID = time*6 + 0
            print('bottom ' + str(r) + str(c)+ ' timeID= '+str(time)+', gID= '+str(gID)+',  Y/M/D/H='+str(yearmc[l,s])+','+str(monthmc[l,s])+','+str(daymc[l,s])+','+str(hourmc[l,s]) ) 
        else:
            print("Warning: cyclone location not found\n")
         
        print(gID)
        #Dot_print the pre-saved image.
        
        #First create the image
        np.save('./Image_2D_true/truth_' + str(gID) + '.npy', ground_true)
        plt.savefig('./Image_2D_true/truth_' + str(gID) + '.png')
        
        img = np.load('./Image_2D_true/truth_' + str(gID) + '.npy')
        img = dotImage(img, r, c)
        np.save('./Image_2D_true/truth_' + str(gID) + '.npy', img)
        plt.imshow(img)     
        plt.savefig('./Image_2D_true/truth_' + str(gID) + '.png')      
        plt.show()
        
print(n_cyc)










