#=======================================================================
# The main code that is used for Bayesian Line fit
# Author: Oscar Javier Hernandez
#=======================================================================


import numpy as np
from scipy.integrate import quad,nquad, dblquad
from scipy import integrate

import pylab
import math
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit
from scipy import stats
import random
import sys


class bayesian_line:
    
    Nquad = 200
    
    # The constructor for the class
    def __init__(self,x_data,y_data):
        
        # Initialize the slope
        self.a=0.0
        self.da=1.0
        
        # Initialize the intercept
        self.b=0.0
        self.db=0.0
        
        # Load in the data
        self.x=x_data
        self.y=y_data
        
        self.N= len(x_data)
        
        
    # The value of Chi squared    
    def chi2(self,a,b,sigma):
        s = 0.0
        
        for i in range(0,self.N):
            yi = self.y[i]
            xi = self.x[i]
            s = s + (1.0/sigma**2)*(yi-a*xi-b)**2
        
        
        return s
        
    # The function P(D|a,b)
    def data_likelyhood(self,a,b,sigma):
        
        s = np.exp(-0.5*self.chi2(a,b,sigma)-0.5*self.N*np.log(2.0*np.pi*sigma**2) + self.N*np.log(2))
        
        return s
        
    
    # The total likelyhood function
    def total_likelyhood(self,a,b,sigma):
        
        s = self.data_likelyhood(a,b,sigma)*self.prior_a(a)*self.prior_b(b)
        
        return s
        
    # This function defines the Prior of the slope
    def prior_a(self,a):
        
        s = 1.0
        
        return s
    
    # This function defines the Prior of the intercept
    def prior_b(self,b):
        
        s = 1.0
        
        
        return s
    
    #-------------------------------------------------------------------    
    # Need to define a proposal distribution
    #-------------------------------------------------------------------
    def q(self,a,b,sigma,dsigma):
        y1 = random.gauss(a, dsigma)
        y2 = random.gauss(b, dsigma)
        y3 = random.gauss(sigma,dsigma)
        return y1,y2,abs(y3)
    
    # Define the acceptance criteria
    def r(self,a,b,sig,at,bt,sigt):
        
        if self.total_likelyhood(at,bt,sigt)==0:
            return 1.0
        else:
            ratio = self.total_likelyhood(a,b,sig)/self.total_likelyhood(at,bt,sigt)
           # ratio =  np.exp(0.5*self.chi2(a,b,sig)-0.5*self.chi2(at,bt,sigt)+0.5*self.N*np.log((sigt/sig)**2))
        
        return min(1.0,ratio)
    
    def acceptance(self,a,b,sig,at,bt,sigt):
        # initialize a random number
        u = random.random()
        
        # calculate the min of the ratio
        ratio = self.r(a,b,sig,at,bt,sigt)
        
        if u <= ratio:
            atp1 = a
            btp1 = b
            sigtp1 = sig
        else:
            atp1 = at
            btp1 = bt
            sigtp1 = sigt
        
        return atp1,btp1,sigtp1
    
    # Markov Chain Monte Carlo
    def MCMC(self,sig0,Tmax,Tburn,dsigma):
        """
        a0 = initial conditions for slope
        b0 = initial conditions for intercept
        Tmax = number of Monte Carlo simulations
        dsigma = 
        
        """
        a0, b0, r_value, p_value, std_err = stats.linregress(self.x,self.y)
        
        # The Markov Chains
        A = []
        B = []
        Sig = []
        T = []
        
        am = a0
        bm = b0
        sigm = sig0
        
        for t in range(Tmax):
            
            at = am
            bt = bm
            sigt = sigm
            a,b,sig = self.q(at,bt,sigt,dsigma)
            
            atp1,btp1,sigtp1 = self.acceptance(a,b,sig,at,bt,sigt)
            
            # Now the next point will be saved as the previous one
            am = atp1
            bm = btp1
            sigm = sigtp1
            
            A.append(am)
            B.append(bm)
            Sig.append(sigm)
            T.append(t)
        
        A = np.asarray(A[Tburn:Tmax])
        B = np.asarray(B[Tburn:Tmax])
        Sig = np.asarray(Sig[Tburn:Tmax])
        T = np.asarray(T[Tburn:Tmax])
            
        return A,B,Sig,T
    
    
    def calculate_results(self,sig0,Tmax,Tburn,dsigma):
        
        a0, b0, r_value, p_value, std_err = stats.linregress(self.x,self.y)
        
        A,B,Sig,T = self.MCMC(sig0,Tmax,Tburn,dsigma)
        
        a = A.mean()
        da = A.std()
        b = B.mean()
        db = B.std()
        
        print(Sig.mean())
        print(Sig.std())
        print("")
        
        #Store Values
        self.a = a
        self.da = da
        self.b = b
        self.db = db
        
        
        return a,da,b,db
    
    # Make a prediction of your weight based on the data
    def predict(self,ac,bc,dac,dbc,x):
        
        y_pred = (ac*x)+bc
        dy_pred = np.sqrt((dac*x)**2+(dbc)**2)
        
        return y_pred,dy_pred
    
    # The Normalization constant of the total likelyhood function
    def normalize(self):
        
        s = integrate.nquad(self.total_likelyhood,[[self.a_min,self.a_max],[self.b_min,self.b_max]])[0]
        
        return s
    
    
    # P(a|D,I): The likelyhood of the slope a
    def slope_parameter(self,a):
        
        def kernel(x):
            
            return self.total_likelyhood(a,x)
        
        s = np.log(integrate.fixed_quad(kernel,self.b_min,self.b_max,n=self.Nquad)[0])
        s = s -np.log(self.Normalization)
        s = np.exp(s)
        
        return s
        
     # P(a|D,I): The likelyhood of the slope a
    def intercept_parameter(self,b):
        
        def kernel(x):
            
            return self.total_likelyhood(x,b)
        
        d = integrate.fixed_quad(kernel,self.a_min,self.a_max,n=self.Nquad)[0]
        
        if d == 0:
            s =0.0
        else:
            s = np.log(d)
            s = s - np.log(self.Normalization)
            s = np.exp(s)
        
        return s
        
    # Returns the results of all of the analysis
    def results(self):
        
        def kernel_a(a):
            s =  a*self.slope_parameter(a)
            return s
        
        def kernel_b(b):
            s =  b*self.intercept_parameter(b)
            return s
        
        def kernel_a2(a):
            s =  a*a*self.slope_parameter(a)
            return s
        
        def kernel_b2(b):
            s =  b*b*self.intercept_parameter(b)
            return s
        
        mean_a  = integrate.quad(kernel_a,self.a_min,self.a_max)[0]
        mean_a2  = integrate.quad(kernel_a2,self.a_min,self.a_max)[0]
        
        print("a: "+str(mean_a)+" "+str(mean_a2))
        
        mean_b  = integrate.quad(kernel_b,self.b_min,self.b_max)[0]
        mean_b2  = integrate.quad(kernel_b2,self.b_min,self.b_max)[0]
        
        print("b: "+str(mean_b)+" "+str(mean_b2))
        
        dev_a = np.sqrt(mean_a2-mean_a**2)
        dev_b = np.sqrt(mean_b2-mean_b**2)
        
        return mean_a,dev_a,mean_b, dev_b
    
    # P(b|D,I): The likelyhood of the intercept b
    
    
    
    
        
    
    
