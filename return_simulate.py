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

        x = rv.rvs(select_num)

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

    simul = Simulation(ret_df)
    x = simul.monte_carlo_simulation()

    ef = EfficientFrontier(ret_df, 0.002, 252)










