import pdb
import seaborn
import numpy as np
import matplotlib.pyplot as plt
import kernels
import gaussianProcess as GP



class smellScape:
    def __init__(self, init_data):

        # read in the initial data
        idata = np.genfromtxt(init_data, delimiter=' ', names=True)
        self.x = idata['x']
        self.y = idata['y']
        self.sensor = idata['sensor']
        self.noise = np.zeros(len(self.sensor))

        # define field of play
        self.xlim = [-1, 1]
        self.ylim = [-1, 1]


    def generate_data_request(self, nx, ny, xlim, ylim, suffix=None):
        
        x_request = np.linspace(xlim[0], xlim[1], nx)
        y_request = np.linspace(ylim[0], ylim[1], ny)
        xy_request = np.meshgrid(x_request, y_request)
        pairs = np.array([np.ravel(xy_request[0]), np.ravel(xy_request[1])]).T
        np.savetxt('request{}.dat'.format(suffix), pairs, fmt='%.4f')
   

    def add_data(self, data):
        
        # read in the new data
        data = np.genfromtxt(data, delimiter=' ', names=True)
        self.x = np.hstack([self.x, data['x']])
        self.y = np.hstack([self.y, data['y']])
        self.sensor = np.hstack([self.sensor, data['sensor']])
        self.noise = np.zeros(len(self.sensor))
        

    def view_field(self):
        
        f = plt.figure()
        ax = f.add_subplot(111)
        sc = ax.scatter(self.x, self.y, marker='^', c=self.sensor, cmap=plt.cm.viridis, s=200)
        plt.colorbar(sc)
        ax.set_xlim(self.xlim)
        ax.set_ylim(self.ylim)
        plt.show()


    def run_gp_regression(self, n_pred, tau, l):
        
        
        xgrid = np.linspace(self.xlim[0], self.xlim[1], n_pred)
        ygrid = np.linspace(self.ylim[0], self.ylim[1], n_pred)
        xygrid = np.meshgrid(xgrid, ygrid)
        pred_points = np.array([np.ravel(xygrid[0]), np.ravel(xygrid[1])]).T

        input_points = np.array([self.x, self.y]).T

        
        mean_func = lambda x: np.zeros((x.shape[0]))
        kernel = lambda x,xp : kernels.sqExpNd(x, xp, tau=tau, l=l)
        GP.run_gp_regression(mean_func, kernel, input_points, self.sensor, pred_points, self.noise)
        
        


if __name__ == '__main__':
    s = smellScape('initialReadings.csv')
    s.generate_data_request(4, 3, [-0.8, 0.8], [0, -0.8], '2')
    s.add_data('sensor_request1.dat')
    
    ll = np.linspace(-4, 1, 10)
    for l in ll:
        s.run_gp_regression(20, 1, 10**l)
    
        

        
