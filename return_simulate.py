import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

class Simulation:

    def __init__(self, ret_df:pd.DataFrame):
        self.ret_df = ret_df

    def monte_carlo_simulation(self):

        '''
        :return:
        '''
        estimated_mean = self.ret_df.mean()
        estimated_cov = self.ret_df.cov()

        get_return_num =

