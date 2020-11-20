import numpy as np
import matplotlib.pyplot as plt
from scipy import special

# ================================================================================
# ================================================================================

class Pareto:
    def __init__(self, alpha, xm=1):
        '''
        Defines a Pareto distribution by the parameter alpha

        Parameters
        ----------
        alpha : float
            The power value parameter
        xm : float, optional
            The scale parameter, defaults to 1

        Attributes
        ----------
        alpha : float
            The power value parameter
        xm : float
            The scale parameter
        max : float
            The location of maximum probability
        mean : float
            The mean of the distribution
        variance : float
            The variance of the distribution

        Methods
        -------
        probability_distribution()
            Returns the PDF for the real number x
        cumulative_distribution()
            Returns the CDF for the real number x
        inverse_cumulative_distribution()
            Returns the inverse CDF for the real number u
        '''

        self.alpha = alpha
        self.xm = xm
        self.max = xm
        
        if(alpha <= 1): self.mean = float('inf')
        else: self.mean = alpha*xm / (alpha-1)
        
        if(alpha <= 2): self.variance = float('inf')
        else: self.variance = (xm**2 * alpha) / ((alpha-1)**2 * (alpha-1))

    # ---------------------------------------------------------------------

    def probability_density(self, x):
        '''
        Evaluate the probability density function for the real number x
        
        Parameters
        ----------
        x : float array-like
            Locations to evaluate the pdf

        Returns
        -------
        pX : float array
            the pdf at each of the input locations
        '''
        
        x = np.array(x)
        pX = np.zeros(len(x))
        pX[x >= self.xm] = (self.alpha * self.xm**self.alpha) / x[x >= self.xm]**(self.alpha+1)
        pX[x < self.xm] = 0
        return pX
     
    # ---------------------------------------------------------------------

    def cumulative_density(self, x):
        '''
        Evaluate the cumulative density function for the real number x
        
        Parameters
        ----------
        x : float array-like
            Locations to evaluate the cdf

        Returns
        -------
        FX : float array
            the cdf at each of the input locations
        '''
        
        FX = np.zeros(len(x))
        FX[x >= self.xm] = 1 - (self.xm / x[x >= self.xm])**self.alpha
        FX[x < self.xm] = 0
        return FX
    
    
    # ---------------------------------------------------------------------

    def inverse_cumulative_density(self, u):
        '''
        Evaluate the inverse cumulative density function for the real number u
        
        Parameters
        ----------
        u : float array-like
            Locations to evaluate the inverse cdf

        Returns
        -------
        iFX : float array
             the inverse cdf at each of the input locations
        '''
        u = np.array(u) 
        iFX = np.zeros(len(u))
        iFX = self.xm / ((1-u)**(1/self.alpha))
        return iFX




# ================================================================================
# ================================================================================
    

class Gaussian():
    def __init__(self, mu=0, sigma=1):
        '''
        Defines a Gaussian distribution by the parameter alpha

        Parameters
        ----------
        mu : float, optional
            The location parameter, defaults to zero
        sigma : float, optional
            The scale parameter, defaults to 1

        Attributes
        ----------
        max : float
            The location of maximum probability
        mean : float
            The mean of the distribution
        variance : float
            The variance of the distribution

        Methods
        -------
        probability_distribution()
            Returns the PDF for the real number x
        cumulative_distribution()
            Returns the CDF for the real number x
        inverse_cumulative_distribution()
            Returns the inverse CDF for the real number u
        '''

        self.mean = mu
        self.max = mu
        self.variance = sigma**2

    # ---------------------------------------------------------------------

    def probability_density(self, x):
        '''
        Evaluate the probability density function for the real number x
        
        Parameters
        ----------
        x : float array-like
            Locations to evaluate the pdf

        Returns
        -------
        pX : float array
            the pdf at each of the input locations
        '''
        
        x = np.array(x)
        sigma = np.sqrt(self.variance)
        pX = 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-(1/2) * (x-self.mean)**2 / sigma**2)
        return pX
     
    # ---------------------------------------------------------------------

    def cumulative_density(self, x):
        '''
        Evaluate the cumulative density function for the real number x
        
        Parameters
        ----------
        x : float array-like
            Locations to evaluate the cdf

        Returns
        -------
        FX : float array
            the cdf at each of the input locations
        '''
        
        x = np.array(x)
        sigma = np.sqrt(self.variance)
        FX = (1/2) * (1 + special.erf( (x - self.mean) / (sigma * np.sqrt(2)) ))
        return FX
    
    # ---------------------------------------------------------------------

    def inverse_cumulative_density(self, u):
        '''
        Evaluate the inverse cumulative density function for the real number u
        
        Parameters
        ----------
        u : float array-like
            Locations to evaluate the inverse cdf

        Returns
        -------
        iFX : float array
             the inverse cdf at each of the input locations
        '''
        u = np.array(u) 
        sigma = np.sqrt(self.variance)
        iFX = sigma * np.sqrt(2) * special.erfinv(2*u - 1) + self.mean
        return iFX
