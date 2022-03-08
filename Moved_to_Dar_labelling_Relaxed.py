import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from scipy.stats import chi2_contingency
from Functions.Database import select
from Functions.Labelling import Create_Spreadsheet
from Functions.Plotting import plot_hist, plot_hist_stacked, PlotOverallGroup, Plot_3D
from Functions.Preprocessing import fill_empty_groups, wide_to_long
from Functions.Stat_tests import t_test, bonferroni_thresh

first_id = 251
last_id = 300

query = "with cte as" \
        "(select distinct customer_id, "\
        "dense_rank() over (order by customer_id) as id_num "\
		"from rosa.moved_to_dar_group_call_data_for_labelling_relaxed) "\
        "select rosa.moved_to_dar_group_call_data_for_labelling_relaxed.*, "\
        "cte.id_num "\
        "from rosa.moved_to_dar_group_call_data_for_labelling_relaxed "\
        "join cte on "\
        "cte.customer_id = rosa.moved_to_dar_group_call_data_for_labelling_relaxed.customer_id "\
        "where cte.id_num between {} and {};".format(first_id, last_id)

if __name__ == '__main__':
    df = select(query)

save_path = "Results/Moved_to_Dar/Labelling/3D_Plots_Relaxed/"

# create spreadsheet:
Create_Spreadsheet(df, first_id, last_id, save_path + "Spreadsheets/")

# plot:
Plot_3D(df, save_path + "Scatter/",
        x='tower_geom_x',
        y='tower_geom_y',
        move_date_col='move_date',
        region_col='tower_region',
        c_dar='m',
        c_not='c',
        c_uni='black',
        type='line',
        n_graphs=(last_id-first_id)+1,
        id_count_col='id_num',
        uni_strict=True
        )

print('done')