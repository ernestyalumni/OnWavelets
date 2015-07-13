## 1dGrid.py
## This is my implementation of Interpolating Wavelets
## starting with a 1d grid
## using Python libraries numpy, scipy
## 
## The main reference that I'll use is
## Oleg V. Vasilyev and Christopher Bowman, Second-Generation Wavelet Collocation Method for the Solution of Partial Differential Equations, Journal of Computational Physics 165, 660-693 (2000)
## 
## 
## 
#####################################################################################
## Copyleft 2015, Ernest Yeung <ernestyalumni@gmail.com>                 
##                                                                                 
## 20150627
##                                                                          
## This program, along with all its code, is free software; 
## you can redistribute it and/or modify  
## it under the terms of the GNU General Public License as published by                
## the Free Software Foundation; either version 2 of the License, or        
## (at your option) any later version.                               
##                                                                
## This program is distributed in the hope that it will be useful,             
## but WITHOUT ANY WARRANTY; without even the implied warranty of                      
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the    
## GNU General Public License for more details.                      
##                                                                       
## You can have received a copy of the GNU General Public License              
## along with this program; if not, write to the Free Software Foundation, Inc.,  
## S1 Franklin Street, Fifth Floor, Boston, MA                      
## 02110-1301, USA                                              
##                                                
## Governing the ethics of using this program, I default to the Caltech Honor Code:  
## ``No member of the Caltech community shall take unfair advantage of               
## any other member of the Caltech community.''                       
##                                                                                  
## If you like what I'm doing and would like to help and contribute support,  
## please take a look at my crowdfunding campaign at ernestyalumni.tilt.com 
## and subscription-based Patreon   
## read my mission statement and give your financial support, 
## no matter how small or large, 
## if you can        
## and to keep checking my ernestyalumni.wordpress.com blog and 
## various social media channels    
## for updates as I try to keep putting out great stuff.                          
##                                                                              
## Fund Science! Help my physics education outreach and research efforts at 
## Open/Tilt or subscription Patreon - Ernest Yeung
##                                                                            
## ernestyalumni.tilt.com                                                         
##                                                                                    
## Facebook     : ernestyalumni                                                       
## gmail        : ernestyalumni                                               
## google       : ernestyalumni                                                    
## linkedin     : ernestyalumni                                                  
## Patreon      : ernestyalumni
## Tilt/Open    : ernestyalumni                                                   
## tumblr       : ernestyalumni                                                       
## twitter      : ernestyalumni                                               
## youtube      : ernestyalumni                                                 
## wordpress    : ernestyalumni                                      
##  
##                                                                           
################################################################################
## 
## 
## 
##
## EY : 20150621 on the MIT OCW website for 18.327, and in the download, phivals.m isn't even a text file; it's in html for some reason.  However, on the web.mit.edu website, the formating is correct, although it's still a html file.

import numpy as np
import scipy
from scipy.linalg import toeplitz

from numpy import power

import matplotlib.pyplot as plt


def faireGrid(J,K,K_0=0,J_0=0):
    """
    faireGrid = faireGrid(J,K,J_0,K_0) faire est make
    INPUTS:
    J = maximum number of level of resolutions
    K = maximum number of grid points for a level
    J_0 = lowest level of resolution (default is 0)
    K_0 = lowest grid point (default is 0)
    OUTPUT:
    numpy array of jxk grid points

    e.g. (Example of usage)
    grid = faireGrid(4,5)

    grid = faireGrid(4,10)

    [ [k for k in grid[j] if k <= grid[-1][-1] ] for j in range(grid.shape[0]) ]

    """
    return np.array( [ [ k/power(2.0,j) for k in range(K_0, K+1)] for j in range(J_0,J+1) ] )

# We can truncate this grid to the maximum value of the x-coordinate for the highest resolution J

def trunc_grid(grid):
    return np.array( [[k for k in grid[j] if (k <= grid[-1][-1] and k >= grid[-1][0])] for j in range(grid.shape[0]) ])


def interpolatePoly(x,y,N):
    """
    interpolatePoly = interpolatePoly(x,y,N)
    """
    assert len(x) == 2*N and len(y) == 2*N

    order = 2*N-1

    return np.polyfit(x,y,order)


def forward_interpolating_wavelet_transform(x,y,N): 
    """
    forward_interpolating_wavelet_transform = forward_interpolating_wavelet_transform(x,y)
    go from resolution level j+1 to j 
    
    OUTPUT:
    dj_k, cj_k  # d^j_k, c^j_k wavelet and scaling function coeffients as types 1xlen(G^j), 1xlen(G^j) numpy arrays, where G^j is the grid for the jth resolution level # x[::2] are the "even" grid points

    TEST VALUES:
    grid_eg2 = faireGrid(5,1000,-1000)
    x_eg2 = trunc_grid( grid_eg2*(0.35/grid_eg2[-1][-1] ))
    x = x_eg2[-1] # grab the highest resolution J as a try with index -1
    y = map(f,x)
    forward_interpolating_wavelet_transform(x,y, 3)
    """
     
    assert 2*N < len(x) and N > 0 
    assert len(x) == len(y)
    # next two immediate lines of code is to handle the polynomial interpolation at the ends of our 1-dimensional line, because we run out of points out to the ends for polynomial interpolation

    N_0 = len(x)/2

    dj_k = []
    for k in range(N_0):
        if k < N-1:
            dj_k.append( np.poly1d( interpolatePoly( x[::2][:2*N], y[::2][:2*N], N) )(x[1::2][k]) )
        elif k > N_0-1 - N:
            dj_k.append( np.poly1d( interpolatePoly( x[::2][-2*N:], y[::2][-2*N:],N))(x[1::2][k]))
        else: 
            dj_k.append( np.poly1d( interpolatePoly( x[::2][k-N+1:k+N+1], y[::2][k-N+1:k+N+1],N))(x[1::2][k] ))
    dj_k = 0.5*(y[1::2] - np.array(dj_k))

    cj_k = y[::2]  # c^j_k

    return dj_k, cj_k

def inverse_wavelet_interpolating_transform(x,d,c, N): 
    """
    inverse_interpolating_wavelet_transform = inverse_interpolating_wavelet_transform(x,d,c)
    go from resolution level j to j+1 
    INPUTS:
    x is grid G^{j+1} of the j+1th resolution level
    
    OUTPUT:
    cjplus1_k # c^{j+1}_k wavelet  coefficients as types 1xlen(G^{j+1}) numpy arrays, where G^{j+1} is the grid for the j+1th resolution level

    TEST VALUES:
    grid_eg2 = faireGrid(5,1000,-1000)
    x_eg2 = trunc_grid( grid_eg2*(0.35/grid_eg2[-1][-1] ))
    x = x_eg2[-2] # grab the second highest resolution J as a try with index -2
    y = map(f,x)
    inverse_wavelet_interpolating_transform(x,d,y, 3)
    """

    assert 2*N < len(x) and N > 0 

    cjplus1_even = c

    cjplus1_odd_start =2*d[:N-1] + np.poly1d(interpolatePoly(x[::2][:2*N],c[:2*N],N))(x[1::2][:N-1])
    cjplus1_odd_end =2*d[:N-1] + np.poly1d(interpolatePoly(x[::2][-2*N:],c[-2*N:],N))(x[1::2][-N+1:])

    Gj_len = len(x[::2])
    cjplus1_odd_middle = 2*d[N-1:-N+1] + np.array([ np.poly1d(interpolatePoly(x[::2][k-N+1:k+N+1],c[k-N+1:k+N+1],N))(x[1::2][k]) for k in range(N-1,Gj_len-N) ])
    cjplus1_odd = np.append(np.append(cjplus1_odd_start,cjplus1_odd_middle),cjplus1_odd_end)

    cjplus1 = [ cjplus1_l for pair in zip(cjplus1_even, cjplus1_odd ) for cjplus1_l in pair ] # cf. http://stackoverflow.com/questions/7946798/interleaving-two-lists-in-python-2-2

    if len(cjplus1_even) > len(cjplus1_odd):
        cjplus1 = np.append( cjplus1, cjplus1_even[ len(cjplus1_odd):] )
    elif len(cjplus1_even) < len(cjplus1_odd):
        cjplus1 = np.append( cjplus1, cjplus1_odd[ len(cjplus1_even):] )

    return cjplus1

def thresholding( data , threshold ):
    """
    thresholding
    get data to be ready to plot by removing y-values that aren't above a certain threshold

    e.g. EXAMPLE of USAGE
    d1_ready = thresholding( zip( x[::16][1::2], d1_k) , 10**(-3) )

    """
    return [ pair for pair in data if pair[-1] > threshold ]
        


####################
## Test values
####################


# second example on pp. 667, Vasilyev and Bowman (2000) Second-Generation Wavelet Collocation Method
# Gaussian envelope-modulated since-frequency signal f(x) = \cos{(80\pi x)e^{-64x^2}

def f(x):
    return np.cos(80.*np.pi*x)*np.exp(-64.*np.power(x,2) ) 


def main():
    grid_eg = faireGrid(4,40) # eg is for example
    trunc_grid_eg = trunc_grid(grid_eg)

#    trunc_grid = np.array( [[k for k in grid[j] if k <= grid[-1][-1] ] for j in range(grid.shape[0]) ])
#    plt.scatter( *zip(*[ (k,j) for j in range(trunc_grid.shape[0]) for k in trunc_grid[j]]) )
#    plt.show()
#    plt.scatter()
    
    # for Example 2 in pp. 667 Vasilyev and Bowman (2000)
    grid_eg2 = faireGrid(5,500,-500)
    x_eg2 = trunc_grid( grid_eg2 * (0.35/ grid_eg2[-1][-1] ))  # 0.35 is the scaling for x, we can always scale the x coordinates

    plt.figure(1)

    plt.subplot(2,1,1)
    plt.plot( x_eg2[0], map( f , x_eg2[0]) )
    plt.title("f(x) = $\cos{(80\pi x)}e^{-64 x^2}$ on a grid at $j=0$ resolution (top), $j=3$ res. (bottom) ")
    
    plt.subplot(2,1,2)
    plt.plot( x_eg2[-2], map( f, x_eg2[-2] ) )
    plt.xlim([x_eg2[0][0], x_eg2[0][-1]])
#    plt.title("f(x) = $\cos{(80 \pi x)}e^{-64 x^2}$ on a grid at $j=3$ resolution")

    N = 3
    x = x_eg2[-1]
    y = map(f,x)
    dj_k,cj_k = forward_interpolating_wavelet_transform(x,y,N)

    cjplus1 = inverse_wavelet_interpolating_transform(x,dj_k, cj_k,N)

    plt.figure(2)

    plt.subplot(3,1,1)
    plt.plot( x[1::2], dj_k, marker='o' , markersize=3 )
    plt.title('d^j_k wavelet coefficients for Example 2')

    plt.subplot(3,1,2)
    plt.plot( x[::2], cj_k, marker='o' , markersize=3 )
    plt.title('c^j_k scaling coefficients for Example 2')
    
    plt.subplot(3,1,3)
    plt.plot( x, cjplus1, marker='o' , markersize=3 )
    plt.title('c^{j+1}_k wavelet coefficients for Example 2')

    d4_k, c4_k = forward_interpolating_wavelet_transform(x[::2], cj_k, N )
    d3_k, c3_k = forward_interpolating_wavelet_transform(x[::4], c4_k, N )
    d2_k, c2_k = forward_interpolating_wavelet_transform(x[::8], c3_k, N )
    d1_k, c1_k = forward_interpolating_wavelet_transform(x[::16], c2_k, N)

    # sanity check
    if len(dj_k) != len( x[1::2]): print "Something went wrong with dj_k"
    if len(d4_k) != len( x[::2][1::2]): print "Something went wrong with d4_k"
    if len(d3_k) != len( x[::4][1::2]): print "Something went wrong with d3_k"
    if len(d2_k) != len( x[::8][1::2]): print "Something went wrong with d2_k"
    if len(d1_k) != len( x[::16][1::2]): print "Something went wrong with d1_k"

    dj_ready = thresholding( zip( x[1::2], dj_k ), 10**(-6))
    d4_ready = thresholding( zip( x[::2][1::2], d4_k ), 10**(-4))
    d3_ready = thresholding( zip( x[::4][1::2], d3_k ), 10**(-3))
    d2_ready = thresholding( zip( x[::8][1::2], d2_k ), 10**(-3))
    d1_ready = thresholding( zip( x[::16][1::2], d1_k ), 10**(-3))

    plt.figure(3)
    plt.subplot(5,1,1)
    plt.scatter(*zip(*dj_ready))
    plt.title("j=5 resolution level")
    plt.ylabel("d^j")

    plt.subplot(5,1,2)
    plt.scatter(*zip(*d4_ready))
    plt.title("j=4 resolution level")
    plt.ylabel("d^j")

    plt.subplot(5,1,3)
    plt.scatter(*zip(*d3_ready))
    plt.title("j=3 resolution level")
    plt.ylabel("d^j")

    plt.subplot(5,1,4)
    plt.scatter(*zip(*d2_ready))
    plt.title("j=2 resolution level")
    plt.ylabel("d^j")

    plt.subplot(5,1,5)
    plt.scatter(*zip(*d1_ready))
    plt.title("j=1 resolution level")
    plt.ylabel("d^j")


    return x,y,dj_k,cj_k,d4_k,c4_k,d3_k,c3_k,d2_k,c2_k,d1_k,c1_k


#    plt.figure(3)
    
#forward_interpolating_wavelet_transform(x,y, N)

if __name__ == "__main__":
    main()
    

