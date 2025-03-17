#%%
import pandas as pd
import numpy as np
import plotly.express as px

#%%
behavior_data = pd.read_csv("data/MINDlarge_train/behaviors.tsv", header=None, sep='\t')
behavior_data.columns = ['impression_id', 'user_id', 'timestamp', 'history', 'impressions']


#%%

behavior_data