import numpy as np
import math
import os
import matplotlib.pyplot as plt


"""
Script for Calculating bending of simply supported plates using double 
Fourier series 

Version June 2022

Luis Santos <pinhosl3@lsbu.ac.uk>
School of Buil Environment and Architecture
London South Bank University 
London, UK

This script was developed for presentation only. Convergence of the results 
are considered for all points simultaneoursly to generate smooth images. 
For a faster script, convergence should be considered point-by-point.
"""


def CombineMoments_vonMises(Mx,My,Mxy):
    
    FOURRIER_Med = np.zeros((np.size(Mx,0),np.size(Mx,1)))
    FOURRIER_Med[0,1:] = Mx[0,1:]
    FOURRIER_Med[1:,0] = Mx[1:,0]    
    FOURRIER_Med[1:,1:] = (Mx[1:,1:]**2-Mx[1:,1:]*My[1:,1:]+My[1:,1:]**2+3*Mxy[1:,1:]**2)**(1/2)
          
    return FOURRIER_Med

def DISPBendingAndShear_Fourier_UDL(L,W,dlx,dly,q):
    
    disp_b = np.zeros((dly+2,dlx+2))
    disp_s = np.zeros((dly+2,dlx+2))
    
    for x in range(0,dlx+1):
        for y in range(0,dly+1):
            disp_b[0,0] = float('nan')
            disp_b[0,x+1] = x*L/dlx
            disp_b[y+1,0] = y*W/dly
            disp_s[0,0] = float('nan')
            disp_s[0,x+1] = x*L/dlx
            disp_s[y+1,0] = y*W/dly
 
    v = 0.3
    
    for xx in range(0,dlx+1):
        for yy in range(0,dly+1):
            sumdisp_b = 0
            sumdisp_s = 0
            x = xx*L/dlx
            y = yy*W/dly
            for m in range(1,100,2):
                for n in range(1,100,2):
                    pmn = 16*q/(math.pi**2*m*n)
                    sumdisp_b = sumdisp_b + pmn*(1/((m/L)**2+(n/W)**2)**2)*math.sin(m*math.pi*x/L)*math.sin(n*math.pi*y/W)
                    sumdisp_s = sumdisp_s + pmn*(1/((m/L)**2+(n/W)**2))*math.sin(m*math.pi*x/L)*math.sin(n*math.pi*y/W)
            disp_b[yy+1, xx+1]= sumdisp_b*(1-v**2)/(math.pi**4)
            disp_s[yy+1, xx+1]= sumdisp_s/(math.pi**2)
            
    print("Done")
    return disp_b, disp_s


def DISPBendingAndShear_Fourier_CL(L,W,dlx,dly,eta,qsi,uu,vv,q):
    
    disp_b = np.zeros((dly+2,dlx+2))
    disp_s = np.zeros((dly+2,dlx+2))
    
    for x in range(0,dlx+1):
        for y in range(0,dly+1):
            disp_b[0,0] = float('nan')
            disp_b[0,x+1] = x*L/dlx
            disp_b[y+1,0] = y*W/dly
            disp_s[0,0] = float('nan')
            disp_s[0,x+1] = x*L/dlx
            disp_s[y+1,0] = y*W/dly
 
    v = 0.3
    
    for xx in range(0,dlx+1):
        for yy in range(0,dly+1):
            sumdisp_b = 0
            sumdisp_s = 0
            x = xx*L/dlx
            y = yy*W/dly
            for m in range(1,50,1):
                for n in range(1,50,1):
                    pmn = 16*q/(math.pi**2*m*n)*math.sin(m*math.pi*qsi/L)*math.sin(n*math.pi*eta/W)*math.sin(m*math.pi*uu/(2*L))*math.sin(n*math.pi*vv/(2*W))
                    sumdisp_b = sumdisp_b + pmn*(1/((m/L)**2+(n/W)**2)**2)*math.sin(m*math.pi*x/L)*math.sin(n*math.pi*y/W)
                    sumdisp_s = sumdisp_s + pmn*(1/((m/L)**2+(n/W)**2))*math.sin(m*math.pi*x/L)*math.sin(n*math.pi*y/W)
            disp_b[yy+1, xx+1]= sumdisp_b*(1-v**2)/(math.pi**4)
            disp_s[yy+1, xx+1]= sumdisp_s/(math.pi**2)
            
    print("Done")
    return disp_b, disp_s


def UDL_Mx_v2(term, Mx0, L, W, dlx, dly, q):
    v = 0.3
    Mx1 = np.copy(Mx0)
    if term == 1:
        n = term
        m = term
        pmn = 16*q/(math.pi**2*m*n)
        for xx in range(0,dlx+1):
            for yy in range(0,dly+1):
                x = xx*L/dlx
                y = yy*W/dly
                Mx1[yy+1,xx+1] = Mx1[yy+1,xx+1] + pmn*(((m/L)**2+v*(n/W)**2)/((m/L)**2+(n/W)**2)**2)*math.sin(m*math.pi*x/L)*math.sin(n*math.pi*y/W)/(math.pi**2)
    else:
        m = term
        n = term
        pmn = 16*q/(math.pi**2*m*n)
        for xx in range(0,dlx+1):
            for yy in range(0,dly+1):
                x = xx*L/dlx
                y = yy*W/dly
                Mx1[yy+1,xx+1] = Mx1[yy+1,xx+1] + pmn*(((m/L)**2+v*(n/W)**2)/((m/L)**2+(n/W)**2)**2)*math.sin(m*math.pi*x/L)*math.sin(n*math.pi*y/W)/(math.pi**2)
        for i in range(1, term, 2):
            m = term
            n = i
            pmn = 16*q/(math.pi**2*m*n)
            for xx in range(0,dlx+1):
                for yy in range(0,dly+1):
                    x = xx*L/dlx
                    y = yy*W/dly
                    Mx1[yy+1,xx+1] = Mx1[yy+1,xx+1] + pmn*(((m/L)**2+v*(n/W)**2)/((m/L)**2+(n/W)**2)**2)*math.sin(m*math.pi*x/L)*math.sin(n*math.pi*y/W)/(math.pi**2)
            m = i
            n = term
            for xx in range(0,dlx+1):
                for yy in range(0,dly+1):
                    x = xx*L/dlx
                    y = yy*W/dly
                    Mx1[yy+1,xx+1] = Mx1[yy+1,xx+1] + pmn*(((m/L)**2+v*(n/W)**2)/((m/L)**2+(n/W)**2)**2)*math.sin(m*math.pi*x/L)*math.sin(n*math.pi*y/W)/(math.pi**2)
    return Mx1


def UDL_My_v2(term, My0, L, W, dlx, dly, q):
    v = 0.3
    My1 = np.copy(My0)
    if term == 1:
        n = term
        m = term
        pmn = 16*q/(math.pi**2*m*n)
        for xx in range(0,dlx+1):
            for yy in range(0,dly+1):
                x = xx*L/dlx
                y = yy*W/dly
                My1[yy+1,xx+1] = My1[yy+1,xx+1] + pmn*(((n/W)**2+v*(m/L)**2)/((m/L)**2+(n/W)**2)**2)*math.sin(m*math.pi*x/L)*math.sin(n*math.pi*y/W)/(math.pi**2)
    else:
        m = term
        n = term
        pmn = 16*q/(math.pi**2*m*n)
        for xx in range(0,dlx+1):
            for yy in range(0,dly+1):
                x = xx*L/dlx
                y = yy*W/dly
                My1[yy+1,xx+1] = My1[yy+1,xx+1] + pmn*(((n/W)**2+v*(m/L)**2)/((m/L)**2+(n/W)**2)**2)*math.sin(m*math.pi*x/L)*math.sin(n*math.pi*y/W)/(math.pi**2)
        for i in range(1, term, 2):
            m = term
            n = i
            pmn = 16*q/(math.pi**2*m*n)
            for xx in range(0,dlx+1):
                for yy in range(0,dly+1):
                    x = xx*L/dlx
                    y = yy*W/dly
                    My1[yy+1,xx+1] = My1[yy+1,xx+1] + pmn*(((n/W)**2+v*(m/L)**2)/((m/L)**2+(n/W)**2)**2)*math.sin(m*math.pi*x/L)*math.sin(n*math.pi*y/W)/(math.pi**2)
            m = i
            n = term
            for xx in range(0,dlx+1):
                for yy in range(0,dly+1):
                    x = xx*L/dlx
                    y = yy*W/dly
                    My1[yy+1,xx+1] = My1[yy+1,xx+1] + pmn*(((n/W)**2+v*(m/L)**2)/((m/L)**2+(n/W)**2)**2)*math.sin(m*math.pi*x/L)*math.sin(n*math.pi*y/W)/(math.pi**2)
    return My1


def UDL_Mxy_v2(term, Mxy0, L, W, dlx, dly, q):
    v = 0.3
    Mxy1 = np.copy(Mxy0)
    if term == 1:
        n = term
        m = term
        pmn = 16*q/(math.pi**2*m*n)
        for xx in range(0,dlx+1):
            for yy in range(0,dly+1):
                x = xx*L/dlx
                y = yy*W/dly
                Mxy1[yy+1,xx+1] = Mxy1[yy+1,xx+1] + pmn*((m/L)*(n/W)/((m/L)**2+(n/W)**2)**2)*math.cos(m*math.pi*x/L)*math.cos(n*math.pi*y/W)*(-1)*(1-v)/(math.pi**2)
    else:
        m = term
        n = term
        pmn = 16*q/(math.pi**2*m*n)
        for xx in range(0,dlx+1):
            for yy in range(0,dly+1):
                x = xx*L/dlx
                y = yy*W/dly
                Mxy1[yy+1,xx+1] = Mxy1[yy+1,xx+1] + pmn*((m/L)*(n/W)/((m/L)**2+(n/W)**2)**2)*math.cos(m*math.pi*x/L)*math.cos(n*math.pi*y/W)*(-1)*(1-v)/(math.pi**2)
        for i in range(1, term, 2):
            m = term
            n = i
            pmn = 16*q/(math.pi**2*m*n)
            for xx in range(0,dlx+1):
                for yy in range(0,dly+1):
                    x = xx*L/dlx
                    y = yy*W/dly
                    Mxy1[yy+1,xx+1] = Mxy1[yy+1,xx+1] + pmn*((m/L)*(n/W)/((m/L)**2+(n/W)**2)**2)*math.cos(m*math.pi*x/L)*math.cos(n*math.pi*y/W)*(-1)*(1-v)/(math.pi**2)
            m = i
            n = term
            for xx in range(0,dlx+1):
                for yy in range(0,dly+1):
                    x = xx*L/dlx
                    y = yy*W/dly
                    Mxy1[yy+1,xx+1] = Mxy1[yy+1,xx+1] + pmn*((m/L)*(n/W)/((m/L)**2+(n/W)**2)**2)*math.cos(m*math.pi*x/L)*math.cos(n*math.pi*y/W)*(-1)*(1-v)/(math.pi**2)
    return Mxy1


def UDL_Qx_v2(term, Qx0, L, W, dlx, dly, q):

    Qx1 = np.copy(Qx0)
    if term == 1:
        n = term
        m = term
        pmn = 16*q/(math.pi**2*m*n)
        for xx in range(0,dlx+1):
            for yy in range(0,dly+1):
                x = xx*L/dlx
                y = yy*W/dly
                Qx1[yy+1,xx+1] = Qx1[yy+1,xx+1] + pmn*(m/L)/((m/L)**2+(n/W)**2)*math.cos(m*math.pi*x/L)*math.sin(n*math.pi*y/W)/(math.pi)
    else:
        m = term
        n = term
        pmn = 16*q/(math.pi**2*m*n)
        for xx in range(0,dlx+1):
            for yy in range(0,dly+1):
                x = xx*L/dlx
                y = yy*W/dly
                Qx1[yy+1,xx+1] = Qx1[yy+1,xx+1] + pmn*(m/L)/((m/L)**2+(n/W)**2)*math.cos(m*math.pi*x/L)*math.sin(n*math.pi*y/W)/(math.pi)
        for i in range(1, term, 2):
            m = term
            n = i
            pmn = 16*q/(math.pi**2*m*n)
            for xx in range(0,dlx+1):
                for yy in range(0,dly+1):
                    x = xx*L/dlx
                    y = yy*W/dly
                    Qx1[yy+1,xx+1] = Qx1[yy+1,xx+1] + pmn*(m/L)/((m/L)**2+(n/W)**2)*math.cos(m*math.pi*x/L)*math.sin(n*math.pi*y/W)/(math.pi)
            m = i
            n = term
            for xx in range(0,dlx+1):
                for yy in range(0,dly+1):
                    x = xx*L/dlx
                    y = yy*W/dly
                    Qx1[yy+1,xx+1] = Qx1[yy+1,xx+1] + pmn*(m/L)/((m/L)**2+(n/W)**2)*math.cos(m*math.pi*x/L)*math.sin(n*math.pi*y/W)/(math.pi)
    return Qx1


def UDL_Qy_v2(term, Qy0, L, W, dlx, dly, q):

    Qy1 = np.copy(Qy0)
    if term == 1:
        n = term
        m = term
        pmn = 16*q/(math.pi**2*m*n)
        for xx in range(0,dlx+1):
            for yy in range(0,dly+1):
                x = xx*L/dlx
                y = yy*W/dly
                Qy1[yy+1,xx+1] = Qy1[yy+1,xx+1] + pmn*(n/W)/((m/L)**2+(n/W)**2)*math.sin(m*math.pi*x/L)*math.cos(n*math.pi*y/W)/(math.pi)
    else:
        m = term
        n = term
        pmn = 16*q/(math.pi**2*m*n)
        for xx in range(0,dlx+1):
            for yy in range(0,dly+1):
                x = xx*L/dlx
                y = yy*W/dly
                Qy1[yy+1,xx+1] = Qy1[yy+1,xx+1] + pmn*(n/W)/((m/L)**2+(n/W)**2)*math.sin(m*math.pi*x/L)*math.cos(n*math.pi*y/W)/(math.pi)
        for i in range(1, term, 2):
            m = term
            n = i
            pmn = 16*q/(math.pi**2*m*n)
            for xx in range(0,dlx+1):
                for yy in range(0,dly+1):
                    x = xx*L/dlx
                    y = yy*W/dly
                    Qy1[yy+1,xx+1] = Qy1[yy+1,xx+1] + pmn*(n/W)/((m/L)**2+(n/W)**2)*math.sin(m*math.pi*x/L)*math.cos(n*math.pi*y/W)/(math.pi)
            m = i
            n = term
            for xx in range(0,dlx+1):
                for yy in range(0,dly+1):
                    x = xx*L/dlx
                    y = yy*W/dly
                    Qy1[yy+1,xx+1] = Qy1[yy+1,xx+1] + pmn*(n/W)/((m/L)**2+(n/W)**2)*math.sin(m*math.pi*x/L)*math.cos(n*math.pi*y/W)/(math.pi)
    return Qy1


def FULLPlate_Fourier_UDL_v3(L,W,dlx,dly,q):

    Mx0 = np.zeros((dly+2,dlx+2))

    for x in range(0,dlx+1):
        for y in range(0,dly+1):
            Mx0[0,0] = float('nan')
            Mx0[0,x+1] = x*L/dlx
            Mx0[y+1,0] = y*W/dly
            
    My0 = np.copy(Mx0)
    Mxy0 = np.copy(Mx0)
    Qx0 = np.copy(Mx0)
    Qy0 = np.copy(Mx0)

    Mx1 = UDL_Mx_v2(1, Mx0, L, W, dlx, dly, q)
    term = 3
    while np.any(Mx1[1:,1:]-Mx0[1:,1:] > 100):
        Mx0 = np.copy(Mx1)
        Mx1 = UDL_Mx_v2(term, Mx0, L, W, dlx, dly, q)
#        print("%s  %s" %(term, Mx1[26,26]))
        term = term + 2

    
    My1 = UDL_My_v2(1, My0, L, W, dlx, dly, q)
    term = 3
    while np.any(My1[1:,1:] - My0[1:,1:] > 100):
        My0 = np.copy(My1)
        My1 = UDL_My_v2(term, My0, L, W, dlx, dly, q)
#        print("%s  %s" %(term, My1[26,26]))
        term = term + 2

    
    Mxy1 = UDL_Mxy_v2(1, Mxy0, L, W, dlx, dly, q)
    term = 3
    while np.any(Mxy1[1:,1:] - Mxy0[1:,1:] > 100):
        Mxy0 = np.copy(Mxy1)
        Mxy1 = UDL_Mxy_v2(term, Mxy0, L, W, dlx, dly, q)
#        print("%s  %s" %(term, Mxy1[1,1]))
        term = term + 2
        
    
    Qx1 = UDL_Qx_v2(1, Qx0, L, W, dlx, dly, q)
    term = 3
    while np.any(Qx1[1:,1:] - Qx0[1:,1:] > 0.01):
        Qx0 = np.copy(Qx1)
        Qx1 = UDL_Qx_v2(term, Qx0, L, W, dlx, dly, q)
#        print("%s  %s" %(term, Qx1[26,1]))
        term = term + 2
        
    
    Qy1 = UDL_Qy_v2(1, Qy0, L, W, dlx, dly, q)
    term = 3
    while np.any(Qy1[1:,1:] - Qy0[1:,1:] > 0.01):
        Qy0 = np.copy(Qy1)
        Qy1 = UDL_Qy_v2(term, Qy0, L, W, dlx, dly, q)
#        print("%s  %s" %(term, Qy1[1,26]))
        term = term + 2
 
    Getreactions = []

    Getreactions.append([Qy1[0,1:], Qy1[1,1:]*L/dlx, "South"])
    Getreactions.append([Qy1[0,1:], Qy1[dly+1,1:]*L/dlx, "North"])
    Getreactions.append([Qx1[1:,0], Qx1[1:,1]*W/dly, "West"])
    Getreactions.append([Qx1[1:,0], Qx1[1:,dlx+1]*W/dly, "East"])

#    print("Done_FULLPlate_Fourier_UDL")
    
#    print(Getreactions)

    return Mx1, My1, Mxy1, Qx1, Qy1, Getreactions


def CL_Mx_v2(term, Mx0, L,W,dlx,dly,eta,qsi,uu,vv,q):
    v = 0.3
    Mx1 = np.copy(Mx0)
    if term == 1:
        n = term
        m = term
        pmn = 16*q/(math.pi**2*m*n)*math.sin(m*math.pi*qsi/L)*math.sin(n*math.pi*eta/W)*math.sin(m*math.pi*uu/(2*L))*math.sin(n*math.pi*vv/(2*W))
        for xx in range(0,dlx+1):
            for yy in range(0,dly+1):
                x = xx*L/dlx
                y = yy*W/dly
                Mx1[yy+1,xx+1] = Mx1[yy+1,xx+1] + pmn*(((m/L)**2+v*(n/W)**2)/((m/L)**2+(n/W)**2)**2)*math.sin(m*math.pi*x/L)*math.sin(n*math.pi*y/W)/(math.pi**2)
    else:
        m = term
        n = term
        pmn = 16*q/(math.pi**2*m*n)*math.sin(m*math.pi*qsi/L)*math.sin(n*math.pi*eta/W)*math.sin(m*math.pi*uu/(2*L))*math.sin(n*math.pi*vv/(2*W))
        for xx in range(0,dlx+1):
            for yy in range(0,dly+1):
                x = xx*L/dlx
                y = yy*W/dly
                Mx1[yy+1,xx+1] = Mx1[yy+1,xx+1] + pmn*(((m/L)**2+v*(n/W)**2)/((m/L)**2+(n/W)**2)**2)*math.sin(m*math.pi*x/L)*math.sin(n*math.pi*y/W)/(math.pi**2)
        for i in range(1, term):
            m = term
            n = i
            pmn = 16*q/(math.pi**2*m*n)*math.sin(m*math.pi*qsi/L)*math.sin(n*math.pi*eta/W)*math.sin(m*math.pi*uu/(2*L))*math.sin(n*math.pi*vv/(2*W))
            for xx in range(0,dlx+1):
                for yy in range(0,dly+1):
                    x = xx*L/dlx
                    y = yy*W/dly
                    Mx1[yy+1,xx+1] = Mx1[yy+1,xx+1] + pmn*(((m/L)**2+v*(n/W)**2)/((m/L)**2+(n/W)**2)**2)*math.sin(m*math.pi*x/L)*math.sin(n*math.pi*y/W)/(math.pi**2)
            m = i
            n = term
            pmn = 16*q/(math.pi**2*m*n)*math.sin(m*math.pi*qsi/L)*math.sin(n*math.pi*eta/W)*math.sin(m*math.pi*uu/(2*L))*math.sin(n*math.pi*vv/(2*W))
            for xx in range(0,dlx+1):
                for yy in range(0,dly+1):
                    x = xx*L/dlx
                    y = yy*W/dly
                    Mx1[yy+1,xx+1] = Mx1[yy+1,xx+1] + pmn*(((m/L)**2+v*(n/W)**2)/((m/L)**2+(n/W)**2)**2)*math.sin(m*math.pi*x/L)*math.sin(n*math.pi*y/W)/(math.pi**2)
    return Mx1


def CL_My_v2(term, My0, L,W,dlx,dly,eta,qsi,uu,vv,q):
    v = 0.3
    My1 = np.copy(My0)
    if term == 1:
        n = term
        m = term
        pmn = 16*q/(math.pi**2*m*n)*math.sin(m*math.pi*qsi/L)*math.sin(n*math.pi*eta/W)*math.sin(m*math.pi*uu/(2*L))*math.sin(n*math.pi*vv/(2*W))
        for xx in range(0,dlx+1):
            for yy in range(0,dly+1):
                x = xx*L/dlx
                y = yy*W/dly
                My1[yy+1,xx+1] = My1[yy+1,xx+1] + pmn*(((n/W)**2+v*(m/L)**2)/((m/L)**2+(n/W)**2)**2)*math.sin(m*math.pi*x/L)*math.sin(n*math.pi*y/W)/(math.pi**2)
    else:
        m = term
        n = term
        pmn = 16*q/(math.pi**2*m*n)*math.sin(m*math.pi*qsi/L)*math.sin(n*math.pi*eta/W)*math.sin(m*math.pi*uu/(2*L))*math.sin(n*math.pi*vv/(2*W))
        for xx in range(0,dlx+1):
            for yy in range(0,dly+1):
                x = xx*L/dlx
                y = yy*W/dly
                My1[yy+1,xx+1] = My1[yy+1,xx+1] + pmn*(((n/W)**2+v*(m/L)**2)/((m/L)**2+(n/W)**2)**2)*math.sin(m*math.pi*x/L)*math.sin(n*math.pi*y/W)/(math.pi**2)
        for i in range(1, term):
            m = term
            n = i
            pmn = 16*q/(math.pi**2*m*n)*math.sin(m*math.pi*qsi/L)*math.sin(n*math.pi*eta/W)*math.sin(m*math.pi*uu/(2*L))*math.sin(n*math.pi*vv/(2*W))
            for xx in range(0,dlx+1):
                for yy in range(0,dly+1):
                    x = xx*L/dlx
                    y = yy*W/dly
                    My1[yy+1,xx+1] = My1[yy+1,xx+1] + pmn*(((n/W)**2+v*(m/L)**2)/((m/L)**2+(n/W)**2)**2)*math.sin(m*math.pi*x/L)*math.sin(n*math.pi*y/W)/(math.pi**2)
            m = i
            n = term
            pmn = 16*q/(math.pi**2*m*n)*math.sin(m*math.pi*qsi/L)*math.sin(n*math.pi*eta/W)*math.sin(m*math.pi*uu/(2*L))*math.sin(n*math.pi*vv/(2*W))
            for xx in range(0,dlx+1):
                for yy in range(0,dly+1):
                    x = xx*L/dlx
                    y = yy*W/dly
                    My1[yy+1,xx+1] = My1[yy+1,xx+1] + pmn*(((n/W)**2+v*(m/L)**2)/((m/L)**2+(n/W)**2)**2)*math.sin(m*math.pi*x/L)*math.sin(n*math.pi*y/W)/(math.pi**2)
    return My1


def CL_Mxy_v2(term, Mxy0, L,W,dlx,dly,eta,qsi,uu,vv,q):
    v = 0.3
    Mxy1 = np.copy(Mxy0)
    if term == 1:
        n = term
        m = term
        pmn = 16*q/(math.pi**2*m*n)*math.sin(m*math.pi*qsi/L)*math.sin(n*math.pi*eta/W)*math.sin(m*math.pi*uu/(2*L))*math.sin(n*math.pi*vv/(2*W))
        for xx in range(0,dlx+1):
            for yy in range(0,dly+1):
                x = xx*L/dlx
                y = yy*W/dly
                Mxy1[yy+1,xx+1] = Mxy1[yy+1,xx+1] + pmn*((m/L)*(n/W)/((m/L)**2+(n/W)**2)**2)*math.cos(m*math.pi*x/L)*math.cos(n*math.pi*y/W)*(-1)*(1-v)/(math.pi**2)
    else:
        m = term
        n = term
        pmn = 16*q/(math.pi**2*m*n)*math.sin(m*math.pi*qsi/L)*math.sin(n*math.pi*eta/W)*math.sin(m*math.pi*uu/(2*L))*math.sin(n*math.pi*vv/(2*W))
        for xx in range(0,dlx+1):
            for yy in range(0,dly+1):
                x = xx*L/dlx
                y = yy*W/dly
                Mxy1[yy+1,xx+1] = Mxy1[yy+1,xx+1] + pmn*((m/L)*(n/W)/((m/L)**2+(n/W)**2)**2)*math.cos(m*math.pi*x/L)*math.cos(n*math.pi*y/W)*(-1)*(1-v)/(math.pi**2)
        for i in range(1, term):
            m = term
            n = i
            pmn = 16*q/(math.pi**2*m*n)*math.sin(m*math.pi*qsi/L)*math.sin(n*math.pi*eta/W)*math.sin(m*math.pi*uu/(2*L))*math.sin(n*math.pi*vv/(2*W))
            for xx in range(0,dlx+1):
                for yy in range(0,dly+1):
                    x = xx*L/dlx
                    y = yy*W/dly
                    Mxy1[yy+1,xx+1] = Mxy1[yy+1,xx+1] + pmn*((m/L)*(n/W)/((m/L)**2+(n/W)**2)**2)*math.cos(m*math.pi*x/L)*math.cos(n*math.pi*y/W)*(-1)*(1-v)/(math.pi**2)
            m = i
            n = term
            pmn = 16*q/(math.pi**2*m*n)*math.sin(m*math.pi*qsi/L)*math.sin(n*math.pi*eta/W)*math.sin(m*math.pi*uu/(2*L))*math.sin(n*math.pi*vv/(2*W))
            for xx in range(0,dlx+1):
                for yy in range(0,dly+1):
                    x = xx*L/dlx
                    y = yy*W/dly
                    Mxy1[yy+1,xx+1] = Mxy1[yy+1,xx+1] + pmn*((m/L)*(n/W)/((m/L)**2+(n/W)**2)**2)*math.cos(m*math.pi*x/L)*math.cos(n*math.pi*y/W)*(-1)*(1-v)/(math.pi**2)
    return Mxy1


def CL_Qx_v2(term, Qx0, L,W,dlx,dly,eta,qsi,uu,vv,q):

    Qx1 = np.copy(Qx0)
    if term == 1:
        n = term
        m = term
        pmn = 16*q/(math.pi**2*m*n)*math.sin(m*math.pi*qsi/L)*math.sin(n*math.pi*eta/W)*math.sin(m*math.pi*uu/(2*L))*math.sin(n*math.pi*vv/(2*W))
        for xx in range(0,dlx+1):
            for yy in range(0,dly+1):
                x = xx*L/dlx
                y = yy*W/dly
                Qx1[yy+1,xx+1] = Qx1[yy+1,xx+1] + pmn*(m/L)/((m/L)**2+(n/W)**2)*math.cos(m*math.pi*x/L)*math.sin(n*math.pi*y/W)/(math.pi)
    else:
        m = term
        n = term
        pmn = 16*q/(math.pi**2*m*n)*math.sin(m*math.pi*qsi/L)*math.sin(n*math.pi*eta/W)*math.sin(m*math.pi*uu/(2*L))*math.sin(n*math.pi*vv/(2*W))
        for xx in range(0,dlx+1):
            for yy in range(0,dly+1):
                x = xx*L/dlx
                y = yy*W/dly
                Qx1[yy+1,xx+1] = Qx1[yy+1,xx+1] + pmn*(m/L)/((m/L)**2+(n/W)**2)*math.cos(m*math.pi*x/L)*math.sin(n*math.pi*y/W)/(math.pi)
        for i in range(1, term):
            m = term
            n = i
            pmn = 16*q/(math.pi**2*m*n)*math.sin(m*math.pi*qsi/L)*math.sin(n*math.pi*eta/W)*math.sin(m*math.pi*uu/(2*L))*math.sin(n*math.pi*vv/(2*W))
            for xx in range(0,dlx+1):
                for yy in range(0,dly+1):
                    x = xx*L/dlx
                    y = yy*W/dly
                    Qx1[yy+1,xx+1] = Qx1[yy+1,xx+1] + pmn*(m/L)/((m/L)**2+(n/W)**2)*math.cos(m*math.pi*x/L)*math.sin(n*math.pi*y/W)/(math.pi)
            m = i
            n = term
            pmn = 16*q/(math.pi**2*m*n)*math.sin(m*math.pi*qsi/L)*math.sin(n*math.pi*eta/W)*math.sin(m*math.pi*uu/(2*L))*math.sin(n*math.pi*vv/(2*W))
            for xx in range(0,dlx+1):
                for yy in range(0,dly+1):
                    x = xx*L/dlx
                    y = yy*W/dly
                    Qx1[yy+1,xx+1] = Qx1[yy+1,xx+1] + pmn*(m/L)/((m/L)**2+(n/W)**2)*math.cos(m*math.pi*x/L)*math.sin(n*math.pi*y/W)/(math.pi)
    return Qx1


def CL_Qy_v2(term, Qy0, L,W,dlx,dly,eta,qsi,uu,vv,q):

    Qy1 = np.copy(Qy0)
    if term == 1:
        n = term
        m = term
        pmn = 16*q/(math.pi**2*m*n)*math.sin(m*math.pi*qsi/L)*math.sin(n*math.pi*eta/W)*math.sin(m*math.pi*uu/(2*L))*math.sin(n*math.pi*vv/(2*W))
        for xx in range(0,dlx+1):
            for yy in range(0,dly+1):
                x = xx*L/dlx
                y = yy*W/dly
                Qy1[yy+1,xx+1] = Qy1[yy+1,xx+1] + pmn*(n/W)/((m/L)**2+(n/W)**2)*math.sin(m*math.pi*x/L)*math.cos(n*math.pi*y/W)/(math.pi)
    else:
        m = term
        n = term
        pmn = 16*q/(math.pi**2*m*n)*math.sin(m*math.pi*qsi/L)*math.sin(n*math.pi*eta/W)*math.sin(m*math.pi*uu/(2*L))*math.sin(n*math.pi*vv/(2*W))
        for xx in range(0,dlx+1):
            for yy in range(0,dly+1):
                x = xx*L/dlx
                y = yy*W/dly
                Qy1[yy+1,xx+1] = Qy1[yy+1,xx+1] + pmn*(n/W)/((m/L)**2+(n/W)**2)*math.sin(m*math.pi*x/L)*math.cos(n*math.pi*y/W)/(math.pi)
        for i in range(1, term):
            m = term
            n = i
            pmn = 16*q/(math.pi**2*m*n)*math.sin(m*math.pi*qsi/L)*math.sin(n*math.pi*eta/W)*math.sin(m*math.pi*uu/(2*L))*math.sin(n*math.pi*vv/(2*W))
            for xx in range(0,dlx+1):
                for yy in range(0,dly+1):
                    x = xx*L/dlx
                    y = yy*W/dly
                    Qy1[yy+1,xx+1] = Qy1[yy+1,xx+1] + pmn*(n/W)/((m/L)**2+(n/W)**2)*math.sin(m*math.pi*x/L)*math.cos(n*math.pi*y/W)/(math.pi)
            m = i
            n = term
            pmn = 16*q/(math.pi**2*m*n)*math.sin(m*math.pi*qsi/L)*math.sin(n*math.pi*eta/W)*math.sin(m*math.pi*uu/(2*L))*math.sin(n*math.pi*vv/(2*W))
            for xx in range(0,dlx+1):
                for yy in range(0,dly+1):
                    x = xx*L/dlx
                    y = yy*W/dly
                    Qy1[yy+1,xx+1] = Qy1[yy+1,xx+1] + pmn*(n/W)/((m/L)**2+(n/W)**2)*math.sin(m*math.pi*x/L)*math.cos(n*math.pi*y/W)/(math.pi)
    return Qy1


def FULLPlate_Fourier_CL_v2(L,W,dlx,dly,eta,qsi,uu,vv,q):

    Mx0 = np.zeros((dly+2,dlx+2))

    for x in range(0,dlx+1):
        for y in range(0,dly+1):
            Mx0[0,0] = float('nan')
            Mx0[0,x+1] = x*L/dlx
            Mx0[y+1,0] = y*W/dly
            
    My0 = np.copy(Mx0)
    Mxy0 = np.copy(Mx0)
    Qx0 = np.copy(Mx0)
    Qy0 = np.copy(Mx0)

    Mx2 = CL_Mx_v2(1, Mx0, L,W,dlx,dly,eta,qsi,uu,vv,q)
    term = 2
    while np.any(Mx2[1:,1:] - Mx0[1:,1:] > 500):
        Mx0 = np.copy(Mx2)
        Mx1 = CL_Mx_v2(term, Mx0, L,W,dlx,dly,eta,qsi,uu,vv,q)
#        print("%s  %s" %(term, Mx1[26,26]))
        term = term + 1
        Mx2 = CL_Mx_v2(term, Mx1, L,W,dlx,dly,eta,qsi,uu,vv,q)
#        print("%s  %s" %(term, Mx1[26,26]))
        term = term + 1

    
    My2 = CL_My_v2(1, My0, L,W,dlx,dly,eta,qsi,uu,vv,q)
    term = 2
    while np.any(My2[1:,1:] - My0[1:,1:] > 500):
        My0 = np.copy(My2)
        My1 = CL_My_v2(term, My0, L,W,dlx,dly,eta,qsi,uu,vv,q)
#        print("%s  %s" %(term, My1[26,26]))
        term = term + 1
        My2 = CL_My_v2(term, My1, L,W,dlx,dly,eta,qsi,uu,vv,q)
#        print("%s  %s" %(term, My1[26,26]))
        term = term + 1

    
    Mxy2 = CL_Mxy_v2(1, Mxy0, L,W,dlx,dly,eta,qsi,uu,vv,q)
    term = 2
    while np.any(Mxy2[1:,1:] - Mxy0[1:,1:] > 500):
        Mxy0 = np.copy(Mxy2)
        Mxy1 = CL_Mxy_v2(term, Mxy0, L,W,dlx,dly,eta,qsi,uu,vv,q)
#        print("%s  %s" %(term, Mxy1[1,1]))
        term = term + 1
        Mxy2 = CL_Mxy_v2(term, Mxy1, L,W,dlx,dly,eta,qsi,uu,vv,q)
#        print("%s  %s" %(term, Mxy1[1,1]))
        term = term + 1
        
    
    Qx2 = CL_Qx_v2(1, Qx0, L,W,dlx,dly,eta,qsi,uu,vv,q)
    term = 2
    while np.any(Qx2[1:,1:] - Qx0[1:,1:] > 1):
        Qx0 = np.copy(Qx2)
        Qx1 = CL_Qx_v2(term, Qx0, L,W,dlx,dly,eta,qsi,uu,vv,q)
#        print("%s  %s" %(term, Qx1[26,1]))
        term = term + 1
        Qx2 = CL_Qx_v2(term, Qx1, L,W,dlx,dly,eta,qsi,uu,vv,q)
#        print("%s  %s" %(term, Qx1[26,1]))
        term = term + 1
        
    
    Qy2 = CL_Qy_v2(1, Qy0, L,W,dlx,dly,eta,qsi,uu,vv,q)
    term = 2
    while np.any(Qy2[1:,1:] - Qy0[1:,1:] > 1):
        Qy0 = np.copy(Qy2)
        Qy1 = CL_Qy_v2(term, Qy0, L,W,dlx,dly,eta,qsi,uu,vv,q)
#        print("%s  %s" %(term, Qy1[1,26]))
        term = term + 1
        Qy2 = CL_Qy_v2(term, Qy1, L,W,dlx,dly,eta,qsi,uu,vv,q)
#        print("%s  %s" %(term, Qy1[1,26]))
        term = term + 1
 
    Getreactions = []

    Getreactions.append([Qy2[0,1:],Qy1[1,1:]*L/dlx, "South"])
    Getreactions.append([Qy2[0,1:],Qy1[dly+1,1:]*L/dlx, "North"])
    Getreactions.append([Qx2[1:,0],Qx1[1:,1]*W/dly, "West"])
    Getreactions.append([Qx2[1:,0], Qx1[1:,dlx+1]*W/dly, "East"])

#    print("Done_FULLPlate_Fourier_CL")
    return Mx2, My2, Mxy2, Qx2, Qy2, Getreactions


def Generate_maps(Stress, name, L, W):
    
    fig, ax1 = plt.subplots()
    im = ax1.imshow(Stress[1:,1:], extent=[0, L, 0, W], cmap='jet_r')
#    ax1.set(aspect=1, title='%s' % (name))
#    ax1.set(aspect=1)
    fig.set_size_inches(6*L/W+5, 6*W/L)
    fig.colorbar(im)
    plt.savefig('%s.jpeg' % (name))
    plt.show()


