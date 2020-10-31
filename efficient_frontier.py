import pandas as pd
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import random
import FinanceDataReader as fdr
import scipy.optimize as opt

from typing import *

class EfficientFrontier:
    '''
    Before making resampled efficient frontier line, costruct basic efficient frontier model
    '''

    def __init__(self, ret_df:pd.DataFrame, rf:float, to_yearly:int):
        '''
        :param ret_df: pd.Dataframe, security return for making efficient frontier line
        :param rf: float, risk-free rate
        :param to_yearly: According to return frequency, make ret_df to annual return and standard deviation
        '''
        self.ret_df = ret_df
        self.rf = rf
        self.num_assets = ret_df.shape[1]
        self.to_yearly = to_yearly

    def weight_loader(self):

        '''
        :return: pd.Dataframe, For making opportunity set, make random weight that matched to asset_num
        :description: For visualization, it makes random weight to make opportunity set
        '''

        weight_arr:np.array = np.random.random(self.num_assets)
        weight_df:pd.DataFrame = pd.DataFrame(weight_arr).T
        weight_df = weight_df.apply(lambda x: x / x.sum(), 1)

        return weight_df

    def get_port_summary(self, weight:pd.DataFrame, mean:np.array, cov:np.array) -> tuple(float, float):

        '''
        :param weight: pd.DataFrame, Weight for make portfolio return, volatility
        :param mean: np.array, Each security's average return mean
        :param cov: np.array, Each security's average return covariance
        :return: tuple(float, float), return portfolio's return, volatility
        :description: Calculate portfolio's return, volatility

        '''

        port_ret:pd.DataFrame = np.dot(weight, mean) * self.to_yearly
        port_vol:pd.DataFrame = np.sqrt(np.dot(np.dot(weight, cov), weight.T)) * np.sqrt(self.to_yearly)

        return port_ret, port_vol

    def get_port_vol(self, weight:pd.DataFrame, mean:np.array, cov:np.array) -> float:

        '''
        :param weight: pd.DataFrame, Weight for make portfolio return, volatility
        :param mean: np.array, Each security's average return mean
        :param cov: np.array, Each security's average return covariance
        :return: float, return portfolio's volatility
        :description: Calculate portfolio's volatility
        '''

        return self.get_port_summary(weight, mean, cov)[1]

    def get_sharpe(self, weight:pd.DataFrame, mean:np.array, cov:np.array) -> float:
        '''
        :param weight: pd.DataFrame, Weight for make portfolio return, volatility
        :param mean: np.array, Each security's average return mean
        :param cov: np.array, Each security's average return covariance
        :return: float, return portfolio's Sharpe ratio
        :description: Calculate portfolio's Sharpe ratio
        '''

        port_ret, port_vol = self.get_port_summary(weight, mean, cov)

        return -1 * ((port_ret - self.rf) / port_vol)

    def max_sharpe(self, mean, cov) -> np.array:
        '''
        :param mean: np.array, Each security's average return mean
        :param cov: np.array, Each security's average return covariance
        :return: np.array, return optimal weight maximize sharpe ratio
        :description: Find optimal weight which maximize sharpe ratio
        '''

        args = (mean, cov)
        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bnds = tuple((0, 1) for asset in range(self.num_assets))
        st = self.num_assets * [1 / self.num_assets]
        res = opt.minimize(self.get_sharpe,
                         st,
                         args=args,
                         bounds=bnds,
                         constraints=cons)
        return res['x']

    def min_vol(self, mean, cov) -> np.array:
        '''
        :param mean: np.array, Each security's average return mean
        :param cov: np.array, Each security's average return covariance
        :return: np.array, return optimal weight minimize portfolio's volatility
        :description: Find optimal weight which maximize volatility
        '''
        args = (mean, cov)
        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bnds = tuple((0, 1) for asset in range(self.num_assets))
        st = self.num_assets * [1 / self.num_assets]
        res = opt.minimize(self.get_port_vol,
                         st,
                         args=args,
                         bounds=bnds,
                         constraints=cons)
        return res['x']

if __name__ == '__main__':
    random.seed(0)
    rf = 0.02
    ret_df = pd.concat([fdr.DataReader('KS11')['Change'],
                       fdr.DataReader('SSEC')['Change'],
                       fdr.DataReader('UK100')['Change']], 1).dropna()

    mean, cov = ret_df.mean(), ret_df.cov() # get each security's return mean and covariance

    mean_vol_ls = []
    simul_num = 1000 # The number of opportunity set

    ef = EfficientFrontier(ret_df, rf, 252)

    for i in range(simul_num):
          weight = ef.weight_loader()
          port_ret, port_vol = ef.get_port_summary(weight, mean, cov)
          mean_vol_ls.append((port_ret, port_vol))

    mean_vol_df = pd.DataFrame(mean_vol_ls)
    max_sharpe_port_ret, max_sharpe_port_std = ef.get_port_summary(ef.max_sharpe(mean, cov), mean, cov)
    min_vol_port_ret, min_vol_port_std = ef.get_port_summary(ef.min_vol(mean, cov), mean, cov)

    # plotting session
    plt.figure(figsize=(12, 6))
    plt.scatter(mean_vol_df[1], mean_vol_df[0], c=(mean_vol_df[0] / mean_vol_df[1]), label='Opportunity Set')
    plt.scatter(max_sharpe_port_std, max_sharpe_port_ret, marker='o', color='r', label='Max Sharpe')
    plt.scatter(min_vol_port_std, min_vol_port_ret, marker='o', color='b', label='Min Volatility')
    plt.scatter(0, rf, marker='o', color='g', label='Risk-Free')
    plt.colorbar(label='sharpe')
    plt.legend()
    plt.show()







