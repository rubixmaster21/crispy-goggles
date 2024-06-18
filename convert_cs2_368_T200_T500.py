from scipy.io import netcdf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
import os


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


######################################################################
os.chdir("C:\\Users\\uyy\\STUFF\\work_alex\\Data_Process")


# Load lat lon file and cube sphere mapping 
cs_file = netcdf.NetCDFFile('ne120np4_latlon.100310.nc','r')  #reference file, variable to coordination. 1D 100K/time_stamp
temp = cs_file.variables['lat']
lat = temp[:]*1
temp = cs_file.variables['lon']
lon = temp[:]*1

xyz = latlon2cubesphere(lat, lon)   #Coordination

ind_sides, ind_top, ind_bottom = cubesphere2t(xyz) #ind_sides(1444,359), ind_top(361,361), ind_bottom(361,361)
ind_sides_1 = ind_sides[    :      360,:]
ind_sides_2 = ind_sides[ 361: 361 +360,:]
ind_sides_3 = ind_sides[ 722: 722 +360,:]
ind_sides_4 = ind_sides[1083:1083 +360,:]

#algorithm to: 1, 0.25 deg/pixel, tempest threshold =4, Gaussian sigma = 10~16
#if (find_in(ind_sides)   break down into 4 square images.
#elif (find_in(ind_top)
# ind_bottom



######################################################################
# Load simulation file and visualize t-image
for i in range(60):
    i = i
    filename = '.\\Training_Files\\train' + str(i+1).zfill(3) + '.nc'
    sim_file = netcdf.NetCDFFile(filename,'r')  #7 variables: PSL, T200, T500, UBOT, VBOT, TUQ, TVQ
    #Map out 7 parameters
    temp1 = sim_file.variables['PSL']  #size=(120, 777602)
    PSL = temp1[:]*1
    temp2 = sim_file.variables['T200']  #size=(120, 777602)
    T200 = temp2[:]*1
    temp3 = sim_file.variables['T500']  #size=(120, 777602)
    T500 = temp3[:]*1
    temp4 = sim_file.variables['UBOT']  #size=(120, 777602)
    UBOT = temp4[:]*1
    temp5 = sim_file.variables['VBOT']  #size=(120, 777602)
    VBOT = temp5[:]*1
    #temp6 = sim_file.variables['TUQ']  #size=(120, 777602)
    #TUQ = temp1[:]*1
    #temp7 = sim_file.variables['TVQ']  #size=(120, 777602)
    #TVQ = temp7[:]*1
    
    
    
    ###+++ Dimension could be 7, 368,368
    t_img3D = np.zeros([5,368,368])
    
    totalIdx = 120   #Measurement in a .nc file (30 days x 4 measurement/day)
    for t in range(totalIdx):   #ind_sides(1444,359), ind_top(361,361), ind_bottom(361,361)
                                #size = (361, 361), save them into .npy file. Save it to the 
        #==================================================                                
        # Section order (ind_sides_1, ind_sides_2, ind_sides_3, ind_sides_4, top, bottom)
        #==================================================

        for section in ["_sides_1","_sides_2","_sides_3","_sides_4","_top","_bottom"]:
            nIdx = 6 * (t + totalIdx*i) + 0
            
            #--- PSL
            t_img0 = PSL[t,eval("ind"+section)]          
            t_img  = t_img0[:360,:359]
            t_img3D[0,4:364, 4:363] = (t_img - t_img.min())/(t_img.max() - t_img.min()) #Normalize the matrix into floating [0..1]
            #--- T200
            t_img0 = T200[t,eval("ind"+section)]  
            t_img  = t_img0[:360,:359]
            t_img3D[1,4:364, 4:363] = (t_img - t_img.min())/(t_img.max() - t_img.min()) #Normalize the matrix into floating [0..1]
            #--- T500
            t_img0 = T500[t,eval("ind"+section)]
            t_img  = t_img0[:360,:359]
            t_img3D[2,4:364, 4:363] = (t_img - t_img.min())/(t_img.max() - t_img.min()) #Normalize the matrix into floating [0..1]
            #--- UBOT
            t_img0 = UBOT[t,eval("ind"+section)] 
            t_img  = t_img0[:360,:359]
            t_img3D[3,4:364, 4:363] = (t_img - t_img.min())/(t_img.max() - t_img.min()) #Normalize the matrix into floating [0..1]
            #--- VBOT
            t_img0 = VBOT[t,eval("ind"+section)] 
            t_img  = t_img0[:360,:359]
            t_img3D[4,4:364, 4:363] = (t_img - t_img.min())/(t_img.max() - t_img.min()) #Normalize the matrix into floating [0..1]
            # #--- TUQ
            # t_img0 = TUQ[t,ind_sides_1]  
            # t_img  = t_img0[:360,:359]
            # t_img3D[6,4:364, 4:363] = (t_img - t_img.min())/(t_img.max() - t_img.min()) #Normalize the matrix into floating [0..1]
            # #--- TVQ
            # t_img0 = TVQ[t,ind_sides_1]  
            # t_img  = t_img0[:360,:359]
            # t_img3D[7,4:364, 4:363] = (t_img - t_img.min())/(t_img.max() - t_img.min()) #Normalize the matrix into floating [0..1]
            
            np.save('./image_3D/full/img3d_' + str(nIdx) + '.npy', t_img3D,bbox_inches='tight', dpi=125)
            print('img3d_' + str(nIdx) + '.npy saved  ' + str(np.shape(t_img3D)) + '   t = ' + str(t) + ', #_nc.file = ' + str(i) )
    
            #Save images of T200 and T500, seperately
            plt.imshow(t_img3D[2,:,:])
            plt.savefig('./image_3D_png/T200/img3d_' + str(nIdx) + '_T200.png',bbox_inches='tight', dpi=125)   # Save image of plot, will convert to video
            plt.show()
            plt.imshow(t_img3D[3,:,:])
            plt.savefig('./image_3D_png/T500/img3d_' + str(nIdx) + '_T500.png',bbox_inches='tight', dpi=125)   # Save image of plot, will convert to video
            plt.show()  
            ### PSL, T200, T500, UBOT, VBOT, (TUQ, TVQ)
            fig, axs = plt.subplots(1, 5)
            axs[0].imshow(t_img3D[0,:,:])
            axs[0].axis('off')  # Turn off axis for cleaner look
            axs[0].set_title(str(t) + "_PSL")
            axs[1].imshow(t_img3D[1,:,:])
            axs[1].axis('off')  # Turn off axis for cleaner look
            axs[1].set_title(str(t) + "_T200")
            axs[2].imshow(t_img3D[2,:,:])
            axs[2].axis('off')  # Turn off axis for cleaner look
            axs[2].set_title(str(t) + "_T500")
            axs[3].imshow(t_img3D[3,:,:])
            axs[3].axis('off')  # Turn off axis for cleaner look
            axs[3].set_title(str(t) + "_UBOT")
            axs[4].imshow(t_img3D[4,:,:])
            axs[4].axis('off')  # Turn off axis for cleaner look
            axs[4].set_title(str(t) + "_VBOT")
            plt.savefig('./image_3D_png/full/img3d_' + str(nIdx) + '.png')   # Save image of plot, will convert to video
            plt.show()  
        
#	s_img = T200[t,ind_sides]
#	plt.imshow(np.transpose(s_img))
#	plt.show()

#	b_img = T200[t,ind_bottom]
#	plt.imshow(b_img)

#T200_back[0, ind_top] = t_img

# ax = plt.axes(projection='3d')
# ax.scatter3D(xyz[1:100,0],xyz[1:100,1],xyz[1:100,2])



