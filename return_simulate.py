import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy as sp
from efficient_frontier import *

from data_loader import *

class Simulation:

    def __init__(self, ret_df:pd.DataFrame):
        self.ret_df = ret_df

    def monte_carlo_simulation(self):

        '''
        :return:
        '''
        estimated_mean = self.ret_df.mean()
        estimated_cov = self.ret_df.cov()

        select_num = len(self.ret_df)

        rv = sp.stats.multivariate_normal(estimated_mean, estimated_cov)

        x = rv.

        return x

    def bootstrapped_ret(self):
        pass


class ResampledEfficient:

    def __init__(self, ret_df):
        self.ret_df = ret_df

    def simulate_efficient(self):

        simul = Simulation(self.ret_df)
        simul_num = 1000
        ef = EfficientFrontier(ret_df, 0.002, 252)


        for i in range(simul_num):
            simul_ret_df = simul.monte_carlo_simulation()


if __name__ == '__main__':

    data_loader = DataLoader()
    ret_df = data_loader.get_sample_return()

    simulation = Simulation(ret_df)
    simul_num = 10
    x = simulation.monte_carlo_simulation()
    frontier_y = np.linspace(0, 0.09, 100)

    for simul in range(simul_num):

        ret_df = pd.DataFrame(x)

        ef = EfficientFrontier(ret_df, 0.002, 252)
        frontier_x, frontier_y = ef.get_efficient_frontier(ret_df.mean(), ret_df.cov(), frontier_y)

        plt.plot(frontier_x, frontier_y)

    plt.show()












