import FinanceDataReader as fdr
import pandas as pd

from typing import *

class DataLoader:

    def __init__(self):
        pass

    def get_sample_return(self) -> pd.DataFrame:
        ret_df = pd.concat([fdr.DataReader('KS11')['Change'],
                            fdr.DataReader('SSEC')['Change'],
                            fdr.DataReader('UK100')['Change']], 1).dropna()

        return ret_df

    def get_custom_return_data(self, ticker_ls:List[str]):
        pass