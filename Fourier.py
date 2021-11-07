import numpy as np
import os
import Fourierfunctions as Ff
import matplotlib.pyplot as plt



#%% CALCULATE THE PANEL USING DOUBLE FOURRIER SERIES

L = 5400
W = 9000
dlx = int(30*1.5)
dly = int(50*1.5)

D = 1
Sx = 1
Sy = 1

K1 = Ff.FULLPlate_Fourier_UDL_v3(L,W,dlx,dly,0.02)
K2 = Ff.FULLPlate_Fourier_CL_v2(L,W,dlx,dly,975,975,450,450,2)
K3 = Ff.FULLPlate_Fourier_CL_v2(L,W,dlx,dly,4500,2700,3000,300,1.5)
K4 = Ff.FULLPlate_Fourier_CL_v2(L,W,dlx,dly,7500,1500,900,900,1.5)
K5 = Ff.FULLPlate_Fourier_CL_v2(L,W,dlx,dly,7500,3900,900,900,1.5)
F1 = Ff.DISPBendingAndShear_Fourier_UDL(L,W,dlx,dly,0.02)
F2 = Ff.DISPBendingAndShear_Fourier_CL(L,W,dlx,dly,975,975,450,450,2)
F3 = Ff.DISPBendingAndShear_Fourier_CL(L,W,dlx,dly,4500,2700,3000,300,1.5)
F4 = Ff.DISPBendingAndShear_Fourier_CL(L,W,dlx,dly,7500,1500,900,900,1.5)
F5 = Ff.DISPBendingAndShear_Fourier_CL(L,W,dlx,dly,7500,3900,900,900,1.5)

# K1 = Rf.FULLPlate_Robinson_UDL_v3(L,W,dlx,dly,0.02,D,Sx,Sy)
# K2 = Rf.FULLPlate_Robinson_CL_v2(L,W,dlx,dly,975,975,450,450,2,D,Sx,Sy)
# K3 = Rf.FULLPlate_Robinson_CL_v2(L,W,dlx,dly,4500,2700,3000,300,1.5,D,Sx,Sy)
# K4 = Rf.FULLPlate_Robinson_CL_v2(L,W,dlx,dly,7500,1500,900,900,1.5,D,Sx,Sy)
# K5 = Rf.FULLPlate_Robinson_CL_v2(L,W,dlx,dly,7500,3900,900,900,1.5,D,Sx,Sy)
# F1 = Rf.DISP_Robinson_UDL_v2(L,W,dlx,dly,0.02,D,Sx,Sy)
# F2 = Rf.DISP_Robinson_CL_v2(L,W,dlx,dly,975,975,450,450,2,D,Sx,Sy)
# F3 = Rf.DISP_Robinson_CL_v2(L,W,dlx,dly,4500,2700,3000,300,1.5,D,Sx,Sy)
# F4 = Rf.DISP_Robinson_CL_v2(L,W,dlx,dly,7500,1500,900,900,1.5,D,Sx,Sy)
# F5 = Rf.DISP_Robinson_CL_v2(L,W,dlx,dly,7500,3900,900,900,1.5,D,Sx,Sy)


aux = ["Mx","My","Mxy","Qx","Qy"]
count = 0
for j in aux:
#    globals()["FOURRIER_%s" % (j)] = K3[count]
    globals()["FOURRIER_%s" % (j)] = K1[count]+K2[count]+K3[count]+K4[count]+K5[count]
    count = count +1

aux = ["Mx","My","Mxy","Qx","Qy"]
count = 0          
for j in aux:
    globals()["FOURRIER_%s" % (j)][0,:] = K5[count][0,:]
    globals()["FOURRIER_%s" % (j)][:,0] = K5[count][:,0]
    count = count +1
        
del count,j, aux

#FOURRIER_Disp = F5

#FOURRIER_Disp_b = F5[0]
#FOURRIER_Disp_s = F5[1]
FOURRIER_Disp_b = F1[0]+F2[0]+F3[0]+F4[0]+F5[0]
FOURRIER_Disp_s = F1[1]+F2[1]+F3[1]+F4[1]+F5[1]

#FOURRIER_Disp = F1+F2+F3+F4+F5



#%% COMPARING THE TWO AND PLOTTING GRAPHS 


"PLOT THE REACTIONS OVER THE FOUR SIDES"

#Ff.PlotReactions_4Sides(FOURRIER_Qx, FOURRIER_Qy, L, W, dlx, dly)
    

"PLOT THE INTERNAL STRESS MAPS ON THE FOURRIER PLATE"

aux = ["Mx","My","Mxy","Qx","Qy"]
for j in aux:
    Ff.Generate_maps(globals()["FOURRIER_%s" % (j)], "FOURRIER_%s" % (j), L, W)

    
#"PLOT DISPLACEMENT MAPS"
#Ff.Generate_maps(FOURRIER_Disp, "FOURRIER_Disp", L, W)
#Ff.Generate_maps(FOURRIER_Disp_b, "FOURRIER_Disp_b", L, W)
#Ff.Generate_maps(FOURRIER_Disp_s, "FOURRIER_Disp_s", L, W)

"GET DESIGN INTERNAL FORCES"
Medx = max(FOURRIER_Mx[1:,1:].max(),-FOURRIER_Mx[1:,1:].min())
Medy = max(FOURRIER_My[1:,1:].max(),-FOURRIER_My[1:,1:].min())
Medxy = max(FOURRIER_Mxy[1:,1:].max(),-FOURRIER_Mxy[1:,1:].min())
Vedx = max(FOURRIER_Qx[1:,1:].max(),-FOURRIER_Qx[1:,1:].min())
Vedy = max(FOURRIER_Qy[1:,1:].max(),-FOURRIER_Qy[1:,1:].min())
# Ded = FOURRIER_Disp[1:,1:].max()
Ded = FOURRIER_Disp_b[1:,1:].max()
MedVM = Ff.CombineMoments_vonMises(FOURRIER_Mx,FOURRIER_My,FOURRIER_Mxy)
#Ff.Generate_maps(MedVM, "FOURRIER_MedVM", L, W)
Med = MedVM[1:,1:].max()
Sed = FOURRIER_Disp_s[1:,1:].max()

"Moments for intercellullar buckling"
Medx_Pos = FOURRIER_Mx[1:,1:].max()
Medy_x_Pos = FOURRIER_My[1:,1:][np.where(FOURRIER_Mx[1:,1:]==Medx_Pos)[0][0]][np.where(FOURRIER_Mx[1:,1:]==Medx_Pos)[1][0]]
Medy_Pos = FOURRIER_My[1:,1:].max()
Medx_y_Pos = FOURRIER_Mx[1:,1:][np.where(FOURRIER_My[1:,1:]==Medy_Pos)[0][0]][np.where(FOURRIER_My[1:,1:]==Medy_Pos)[1][0]]
MedIC = ((Medx_Pos, Medy_x_Pos),(Medx_y_Pos, Medy_Pos))

print("Medx - %s" % Medx)
print("Medy - %s" % Medy)
print("Medxy - %s" % Medxy)
print("Vedx - %s" % Vedx)
print("Vedy - %s" % Vedy)
print("Ded - %s" % Ded)
print("Sed - %s" % Sed)
print("Med - %s" % Med)
print("---------")


    