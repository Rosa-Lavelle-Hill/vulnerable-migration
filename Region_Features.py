import datetime as dt
import pandas as pd
import os
import numpy as np
from Functions.Database import select
# --------------------------------------------------------------------------------------------------------
data_path = "Data/Modelling/DV/"

cont_dv = pd.read_csv(data_path+"Continuous_Dependent_Variable.csv")
bin_dv = pd.read_csv(data_path+"Binary_Dependent_Variable_50.csv")
region_iv = pd.read_csv("Data/Regions/region_data.csv")

# IV correlations
cor = round(region_iv.corr(), 2)
cor_save_path = "Results/Descriptives/Correlations/"
cor.to_csv(cor_save_path+"Region_IVs.csv")

print('done!')