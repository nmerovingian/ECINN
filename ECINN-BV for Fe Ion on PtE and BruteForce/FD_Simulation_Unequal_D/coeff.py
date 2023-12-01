import numpy as np



class Coeff(object):
    def __init__(self,n,deltaT,X_grid,K0,alpha,beta,kinetics,mode,dA,dB):
        self.n = n

        self.aA = np.zeros(self.n)
        self.bA = np.zeros(self.n)
        self.cA = np.zeros(self.n)
        self.aB = np.zeros(self.n)
        self.bB = np.zeros(self.n)
        self.cB = np.zeros(self.n)
        self.deltaT = deltaT

        self.X_grid = X_grid

        self.K0 = K0
        self.alpha = alpha
        self.beta = beta
        self.kinetics = kinetics
        self.mode = mode
        self.dA = dA
        self.dB = dB


        self.A_matrix = np.zeros((2*self.n,2*self.n))

    def update(self):
        pass


    def Acal_abc_radial(self,deltaT):
        
        self.aA[0] = 0.0
        self.bA[0] = 0.0
        self.cA[0] = 0.0

        for i in range(1,self.n-1):
            deltaX_m = self.X_grid[i] - self.X_grid[i - 1]
            deltaX_p = self.X_grid[i + 1] - self.X_grid[i]
            self.aA[i] = self.dA*((-(2.0 * deltaT) / (deltaX_m * (deltaX_m + deltaX_p)) + 2.0 / self.X_grid[i] * (deltaT / (deltaX_m + deltaX_p))))
            self.bA[i] = self.dA*(((2.0 * deltaT) / (deltaX_p * (deltaX_m + deltaX_p)) + (2.0 * deltaT) / (deltaX_m * (deltaX_m + deltaX_p)))) + 1.0
            self.cA[i] = self.dA*((-(2.0 * deltaT) / (deltaX_p * (deltaX_m + deltaX_p)) - 2.0 / self.X_grid[i] * (deltaT / (deltaX_m + deltaX_p))))

        self.aA[-1] = 0.0
        self.bA[-1] = 0.0
        self.cA[-1] = 0.0

    
    def Acal_abc_linear(self,deltaT):

        self.aA[0] = 0.0
        self.bA[0] = 0.0
        self.cA[0] = 0.0

        for i in range(1,self.n-1):
            deltaX_m = self.X_grid[i] - self.X_grid[i - 1]
            deltaX_p = self.X_grid[i + 1] - self.X_grid[i]
            self.aA[i] = self.dA*((-2.0*deltaT)/(deltaX_m*(deltaX_m+deltaX_p)))
            self.cA[i] = self.dA*((-2.0*deltaT)/(deltaX_p*(deltaX_m+deltaX_p)))
            self.bA[i] = 1.0-self.aA[i] - self.cA[i]
        self.aA[-1] = 0.0
        self.bA[-1] = 0.0
        self.cA[-1] = 0.0


    def Bcal_abc_radial(self,deltaT):
        self.aB[0] = 0.0
        self.bB[0] = 0.0
        self.cB[0] = 0.0

        for i in range(1,self.n-1):
            deltaX_m = self.X_grid[i] - self.X_grid[i - 1]
            deltaX_p = self.X_grid[i + 1] - self.X_grid[i]
            self.aB[i] = self.dB*((-(2.0 * deltaT) / (deltaX_m * (deltaX_m + deltaX_p)) + 2.0 / self.X_grid[i] * (deltaT / (deltaX_m + deltaX_p))))
            self.bB[i] = self.dB*(((2.0 * deltaT) / (deltaX_p * (deltaX_m + deltaX_p)) + (2.0 * deltaT) / (deltaX_m * (deltaX_m + deltaX_p)))) + 1.0
            self.cB[i] = self.dB*((-(2.0 * deltaT) / (deltaX_p * (deltaX_m + deltaX_p)) - 2.0 / self.X_grid[i] * (deltaT / (deltaX_m + deltaX_p))))

        self.aB[-1] = 0.0
        self.bB[-1] = 0.0
        self.cB[-1] = 0.0

    def Bcal_abc_linear(self,deltaT):

        self.aB[0] = 0.0  
        self.bB[0] = 0.0
        self.cB[0] = 0.0

        for i in range(1,self.n-1):
            deltaX_m = self.X_grid[i] - self.X_grid[i - 1]
            deltaX_p = self.X_grid[i + 1] - self.X_grid[i]
            self.aB[i] = self.dB*((-2.0*deltaT)/(deltaX_m*(deltaX_m+deltaX_p)))
            self.cB[i] = self.dB*((-2.0*deltaT)/(deltaX_p*(deltaX_m+deltaX_p)))
            self.bB[i] = 1.0-self.aB[i] - self.cB[i]
        self.aB[-1] = 0.0
        self.bB[-1] = 0.0
        self.cB[-1] = 0.0


    def Allcalc_abc(self,deltaT):
        if self.mode =='radial':
            self.Acal_abc_radial(deltaT)
            self.Bcal_abc_radial(deltaT)
        elif self.mode =='linear':
            self.Acal_abc_linear(deltaT)
            self.Bcal_abc_linear(deltaT)
        else:
            raise ValueError


    def CalcMatrix(self,Theta):

        if self.kinetics == 'Nernst':
            self.A_matrix[self.n-1,self.n-2] = 0.0
            self.A_matrix[self.n-1,self.n-1] = 1.0
            self.A_matrix[self.n-1,self.n] = 0.0
            
            self.A_matrix[self.n,self.n-2] = -self.dA
            self.A_matrix[self.n,self.n-1] = self.dA
            self.A_matrix[self.n,self.n] = self.dB
            self.A_matrix[self.n,self.n+1] = -self.dB

        elif self.kinetics =='BV':
            X0 = self.X_grid[1]-self.X_grid[0]
            K_red = self.K0*np.exp(-self.alpha*Theta)
            K_ox = self.K0*np.exp(self.beta*Theta)

            self.A_matrix[self.n-1,self.n-2] = -1.0
            self.A_matrix[self.n-1,self.n-1] = 1.0 + X0/self.dA*K_red
            self.A_matrix[self.n-1,self.n] = -X0/self.dA * K_ox

            self.A_matrix[self.n,self.n-1] = - X0/self.dB*K_red
            self.A_matrix[self.n,self.n] = (1.0 + X0/self.dB*K_ox)
            self.A_matrix[self.n,self.n+1] = -1.0
        else:
            raise ValueError
        
        for i in range(self.n-2,0,-1):

            self.A_matrix[i,i-1] = self.cA[(self.n-1)-i]
            self.A_matrix[i,i] = self.bA[(self.n-1)-i]
            self.A_matrix[i,i+1] = self.aA[(self.n-1)-i]


        for i in range(self.n+1,2*self.n-1):
            self.A_matrix[i,i-1] = self.aB[i-self.n]
            self.A_matrix[i,i] = self.bB[i-self.n]
            self.A_matrix[i,i+1] = self.cB[i-self.n]

        self.A_matrix[0,0] = 1.0
        self.A_matrix[0,1] = 0.0
        self.A_matrix[2*self.n-1,2*self.n-1] = 1.0
        self.A_matrix[2*self.n-1,2*self.n-2] = 0.0



        