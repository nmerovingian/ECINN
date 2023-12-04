import numpy as np
import pandas as pd

class Grid(object):
    def __init__(self,n,X_grid,K0,kinetics,alpha,beta,dA,dB):
        self.n = n
        self.X_grid =  X_grid
        self.K0 = K0
        self.kinetics = kinetics
        self.alpha = alpha
        self.beta = beta
        self.dA = dA
        self.dB = dB
        self.conc = np.zeros(2*self.n)
        self.conc_d = np.zeros(2*self.n)

    def init_c(self,A:float,B:float):
        self.conc[:self.n] = A
        self.conc[self.n:] = B
        self.conc_d = self.conc.copy()
    
    def grad(self):
        self.g = - self.dA*(self.conc[self.n-2]-self.conc[self.n-1])/(self.X_grid[1] - self.X_grid[0])

        return self.g

    def update_d(self,Theta,bulk_A,bulk_B):
        self.conc_d = self.conc.copy()


        if self.kinetics == 'Nernst':
            self.conc_d[self.n-1] = 1.0/(1.0+np.exp(-Theta))
            self.conc_d[self.n] = 0.0
        elif self.kinetics =='BV':
            X0 = self.X_grid[1]-self.X_grid[0]
            K_red = self.K0*np.exp(-self.alpha*Theta)
            K_ox = self.K0*np.exp(self.beta*Theta)
            self.conc_d[self.n-1] = 0.0
            self.conc_d[self.n] = 0.0
        else:
            raise ValueError
        self.conc_d[0] = bulk_A
        self.conc_d[2*self.n-1] = bulk_B

    def save_conc_profile(self,file_name):
        df  = pd.DataFrame({'X':self.X_grid,'Conc':self.conc})
        df.to_csv(file_name,index=False)

    