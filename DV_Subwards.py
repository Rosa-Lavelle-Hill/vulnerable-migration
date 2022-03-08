import datetime as dt
import pandas as pd
import os
import numpy as np
from Functions.Database import select
# --------------------------------------------------------------------------------------------------------
# join towers->subwards->DV

# params:
perc = 25
Temeke = True

# get towers->subwards:
from Functions.Plotting import plot_hist

if Temeke == True:
    t = ""
if Temeke == False:
    t = "_No_Temeke"

f = open("SQL/Queries/Select/DV_Subwards{}.txt".format(t))
q = f.read()
if __name__ == '__main__':
    df_towers_subwards = select(q, section="dar_survey")
subwards_tower_path = "Data/Subwards/"
df_towers_subwards.to_csv(subwards_tower_path+"towers_subwards{}.csv".format(t))

# get subwards->DV:
df_dv_subwards = pd.read_csv("Data/Subwards/subward_quality.csv")

# get night_towers Dar:
df_night_towers = pd.read_csv("Data/Dar_Night_Towers/Dar_Night_Towers.csv", index_col=False)

df_towers = df_towers_subwards.merge(df_dv_subwards, left_on='subward_id', right_on='subward.id')
cols = ["tower_id","subward_name", "subward_id", "ward.name", "ward.id", "district", "rank","prop.known","kmeans"]
df_all = df_night_towers[["ID","Dar_Night_Tower"]].merge(df_towers[cols], left_on="Dar_Night_Tower", right_on='tower_id')
df_cont = df_all[["ID","rank"]]

if Temeke == True:
    id_total = len(df_all.ID.unique())
    subward_total = len(df_towers_subwards.subward_name.unique())

if Temeke == False:
    id_total = len(df_all.ID.unique())
    subward_total = len(df_towers_subwards.subward_name.unique())

data_save_path = "Data/Modelling/DV/"
df_cont.to_csv(data_save_path+"Continuous_Dependent_Variable{}.csv".format(t))
# --------------------------------------------------------------------------------------------------------
# Create Binary DV

# calc how many indivs in wards bottom x%:
perclist = [10,15,20,25,30,50]
max_rank = df_dv_subwards['rank'].max()
for p in perclist:
    x = (max_rank/100)*p
    z = max_rank-x
    target_group = df_all["ID"][df_all["rank"] >= round(z, 0)].count()
    print("Percentage: {} = Target (1): {}/{}".format(p, target_group, id_total))

# plot DV hist of how many people in sample in each subward:
save_path="Results/Descriptives/DV/"
plot_hist(save_path=save_path, save_name="Sanple_subward_rank_distribution{}".format(t), x=df_all['rank'], bins=10, title="",
          xlab="Subward ranks by affluence (1 = richest)", ylab="Number of indivdiuals who moved to the subward",
          fig_size=(8,8), fontsize=14)

# create binary variable:
x = (max_rank/100)*perc
z = max_rank-x

df_all['Binary_DV'] = 0
df_all['Binary_DV'][df_all['rank'] >= z] = 1
df_binary = df_all[['ID','Binary_DV']]
df_all.to_csv(data_save_path+"Binary_Dependent_Variable_{}{}.csv".format(perc, t))

plot_hist(save_path=save_path, save_name="Binary_DV_{}perc{}".format(perc, t), x=df_all['Binary_DV'], bins=3, title="",
          xlab="Binary Variable (1= {}% poorest subwards)".format(perc), ylab="Number of indivdiuals",
          fig_size=(8,8), fontsize=14)

print('done!')
