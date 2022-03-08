import datetime as dt
import pandas as pd
import os
import numpy as np
import scipy
import scipy.stats as st
from Functions.Plotting import plot_hist, plot_hist_stacked

start_time = dt.datetime.now()
# load raw features data
data_save_path = "Data/Modelling/Features/"
df = pd.read_csv(data_save_path + "All_features_no_dummy_nas.csv", index_col=False)

# Replace some NA values with 0's when it makes sense:
replace_nas_with_zeros = ['Overall_P2P_Sent_Rec_Normed', 'Before_P2P_Sent_Rec_Normed', 'After_P2P_Sent_Rec_Normed',
                          'Overall_P2P_Sent_Normed', 'Before_P2P_Sent_Normed', 'After_P2P_Sent_Normed',
                          'Overall_P2P_Received_Normed', 'Before_P2P_Received_Normed', 'After_P2P_Received_Normed',
                          'Overall_Cashin_Normed', 'Before_Cashin_Normed', 'After_Cashin_Normed',
                          'Overall_CashOut_Normed', 'Before_CashOut_Normed', 'After_CashOut_Normed',
                          'Overall_UtilityBill_Normed', 'Before_UtilityBill_Normed', 'After_UtilityBill_Normed',
                          'Had_MFS_Before']
df[replace_nas_with_zeros] = df[replace_nas_with_zeros].fillna(0)

# add region features:
region_iv = pd.read_csv("Data/Regions/region_data.csv")
df.reset_index(inplace=True)
df = df.merge(region_iv, left_on = "Region_From", right_on="Region", how="left")
data_save_path = "Data/Modelling/Features/"
df.to_csv(data_save_path + "All_Features_Region_no_dummies.csv")

# create feature as an average of H_dist_Mean and H_dist_Mode
df["H_dist_Average"] = round((df["H_dist_Mean"] + df["H_dist_Mode"])/2,2)

# dummy code Region_From feature:
region_dummy = pd.get_dummies(df['Region_From'], prefix="Region")
df_all = pd.concat([df, region_dummy], axis=1)

# remove original col and a dummy col (so no singularities)" *don't need to drop if only using dt/rf/lasso models
df_all.drop(['Region_From', 'Region', 'index'], inplace=True, axis=1)
df_all.set_index('ID', inplace=True)
df_all.to_csv(data_save_path + "All_Features_Region.csv")

# Group and select features (ignoring received calling features for now due to bias in data):
overall_call_features = ['Overall_Sent_Calls', 'Overall_Call_Entropy', 'Overall_Mean_Calling_Distance',
                         'Overall_Mode_Calling_Distance']

before_call_features = ['Before_Sent_Calls', 'Before_Call_Entropy', 'Before_Mean_Calling_Distance',
                        'Before_Mode_Calling_Distance', 'Before_Perc_of_Calls_to_Dar',
                        'Before_Calls_to_Dar_Ward_Rank', 'Before_Calls_to_Dar_Subward_Rank']

after_call_features = ['After_Sent_Calls', 'After_Call_Entropy', 'After_Mean_Calling_Distance', 'After_Mode_Calling_Distance',
                       'After_Mean_Calling_Distance_Normed', 'After_Perc_Home_Calls', 'After_Number_of_Home_Visits']

overall_mfs_features = ['Overall_Mean_Balance', 'Overall_Money_Sent_And_Rec_Normed','Overall_Money_Received_Normed',
                        'Overall_Money_Sent_Normed', 'Overall_P2P_Sent_Rec_Normed', 'Overall_P2P_Sent_Normed',
                        'Overall_P2P_Received_Normed', 'Overall_Cashin_Normed', 'Overall_CashOut_Normed',
                        'Overall_UtilityBill_Normed']

before_mfs_features = ['Before_Mean_Balance', 'Before_Money_Sent_And_Rec_Normed','Before_Money_Received_Normed',
                       'Before_Money_Sent_Normed', 'Before_P2P_Sent_Rec_Normed','Before_P2P_Sent_Normed',
                       'Before_P2P_Received_Normed','Before_Cashin_Normed', 'Before_CashOut_Normed',
                       'Before_UtilityBill_Normed', 'Had_MFS_Before']

after_mfs_features = ['After_Mean_Balance','After_Money_Sent_And_Rec_Normed', 'After_Money_Received_Normed',
                      'After_Money_Sent_Normed', 'After_P2P_Sent_Rec_Normed', 'After_P2P_Sent_Normed',
                      'After_P2P_Received_Normed', 'After_Cashin_Normed', 'After_CashOut_Normed',
                      'After_UtilityBill_Normed']

dummy_region_features = ['Region_Arusha', 'Region_Dodoma', 'Region_Geita', 'Region_Iringa',
                   'Region_Kagera', 'Region_Katavi', 'Region_Kigoma', 'Region_Kilimanjaro',
                   'Region_Lindi', 'Region_Manyara', 'Region_Mara', 'Region_Mbeya',
                   'Region_Morogoro', 'Region_Mtwara', 'Region_Mwanza', 'Region_Njombe',
                   'Region_Pemba Island', 'Region_Pwani', 'Region_Rukwa', 'Region_Ruvuma',
                   'Region_Shinyanga', 'Region_Simiyu', 'Region_Singida', 'Region_Tabora',
                   'Region_Tanga', 'Region_Zanzibar']

region_features = ["Region_HDI", "Region_Life_Expectancy", "Region_Expected_Years_Schooling", "Region_GDPpc", "Region_NonIncome_HDI",
                   "Region_Ratio_Female_Male_HDI", "Region_Life_Expectancy_Female", "Region_Life_Expectancy_Male", "Region_EYS_Female_Male_Ratio",
                   "Region_Estimate_GDPpc_Female", "Region_Estimate_GDPpc_Male", "Region_Multidim_Poverty_Index", "Region_Incidence_Poverty",
                   "Region_Av_Intensity_Poor", "Region_Population_Vulnerable_To_Poverty", "Region_Population_In_Severe_Poverty", "Region_Standard_of_Living",
                   "Region_Health", "Region_Education", "Region_Immunization_DPT", "Region_Immunization_Measles", "Region_Child_Nutrition_Stunted",
                   "Region_Child_Nutrition_Wasted", "Region_Under_Weight_Birth", "Region_Death_at_Delivery_Facility", "Region_Delivered_Skilled_Provider",
                   "Region_Delivered_Facility", "Region_Antenatal_Visits", "Region_Women_HIV",	"Region_Men_HIV", "Region_Parliament_Female_Male_Ratio",
                   "Region_District_Commissioners_Female_Male_Ratio", "Region_Concillors_Female_Male_Ratio", "Region_Adult_Literacy",
                   "Region_Population_Secondary_Education", "Region_Population_Millions",
                   "Region_Population_0_4years", "Region_Population_5_14years",	"Region_Population_15_64years",	"Region_Population_65years_plus",
                   "Region_Urban_Population", "Region_Population_Density_km2", "Region_Median_Age", "Region_Age_Dependency_Ratio",
                   "Region_Fertility_Rate",	"Region_Male_Female_Ratio"]

other_overall_features = ['H_dist_Average', 'Visited_University']

other_before_features = ['Visited_Dar_Before_Moved', 'Rural_Ward', 'H_Dist_to_Dar_Before']

# select features all:
select_features_all = overall_call_features + before_call_features + after_call_features + overall_mfs_features\
                      + before_mfs_features + after_mfs_features + other_overall_features + other_before_features\
                      + region_features + dummy_region_features
# select features before:
select_features_before = region_features + before_call_features + before_mfs_features + other_before_features + dummy_region_features

# save dfs:
df_features_selected = df_all[select_features_all]
df_features_selected.to_csv(data_save_path+"Selected_Features_Full.csv")

df_before_features_selected = df_all[select_features_before]
df_before_features_selected.to_csv(data_save_path+"Selected_Features_Before.csv")

# ----------------------------------------------------------------------------------------------------------
# plot histograms:

df['Mean_Mode_Error'] = abs(df['H_dist_Mean'] - df['H_dist_Mode'])
high_err_examples = df.sort_values(by="Mean_Mode_Error", ascending=False)[0:5]

save_path = "Results/Descriptives/Features/"

# plot hist
plot_hist_stacked(save_name='Migration_distances', x1=df['H_dist_Mean'], x2=df['H_dist_Mode'],
                  bins=10, save_path=save_path, x1_lab= "Mean", x2_lab= "Mode", ylab="Frequency",
                  title= "Migration distance (Haversine) calulated using two different methods",
                  xlim=None, fig_size=(6,6), title_fontsize=14, xlab="Distance migrated (km)")

r, p = scipy.stats.pearsonr(df['H_dist_Mean'], df["H_dist_Mode"])
print("Pearson r correlation: {:.4f}, p= {:.4f}".format(r,p))

df.set_index('ID', inplace=True)
for col in df.columns:
    if (col != "Region_From") and (col!= "Region"):
        print(col)
        plot_hist(save_name=col, x=df[col],
                  bins=10, save_path=save_path, title= col.replace('_',' '),
                  xlim=None, ylim=None, xlab=col.replace('_',' '), ylab="Frequency",
                  fig_size=(6,6))

runtime = dt.datetime.now() - start_time
print('Total run time: ' + str(runtime))


print('done')