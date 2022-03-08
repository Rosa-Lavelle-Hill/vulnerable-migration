import datetime as dt
import pandas as pd
import os
import numpy as np
import scipy.stats as st
import statsmodels as sm

from Functions.Plotting import Plot_ANOVA
from Functions.Stat_tests import bonferroni_thresh, t_test, Twoway_ANOVA

print('running')

perc = 50
Temeke = False
save_params = True

if Temeke == True:
    t = ""
if Temeke == False:
    t = "_No_Temeke"

X_data_path = "Data/Modelling/Features/"
y_data_path = "Data/Modelling/DV/"

# load IVs
X_all = pd.read_csv(X_data_path + "Selected_Features_Full.csv")
X_all.set_index("ID", inplace=True)

X_before = pd.read_csv(X_data_path + "Selected_Features_Before.csv")
X_before.set_index("ID", inplace=True)

# load DV
y = pd.read_csv(y_data_path + "Binary_Dependent_Variable_{}{}.csv".format(perc, t), index_col=None, usecols=["ID", "Binary_DV"])
y.set_index("ID", inplace=True)

y_c = pd.read_csv(y_data_path + "Continuous_Dependent_Variable.csv", index_col=None, usecols=["ID", "rank"])
y_c.set_index("ID", inplace=True)

# feature selection :

# 1) unsupervised - remove dependencies after looking at correlation matrix visually first

# Reduce selection of features:
before_call_features = ['Before_Sent_Calls', 'Before_Call_Entropy', 'Before_Mode_Calling_Distance',
                         'Before_Perc_of_Calls_to_Dar', 'Before_Calls_to_Dar_Ward_Rank', 'Before_Calls_to_Dar_Subward_Rank']

after_call_features = ['After_Sent_Calls', 'After_Call_Entropy', 'After_Mean_Calling_Distance',
                       'After_Perc_Home_Calls', 'After_Number_of_Home_Visits']

before_mfs_features = ['Before_Mean_Balance','Before_Money_Received_Normed', 'Before_Money_Sent_Normed',
                       'Before_Cashin_Normed', 'Before_CashOut_Normed', 'Before_UtilityBill_Normed', 'Had_MFS_Before']

after_mfs_features = ['After_Mean_Balance', 'After_Money_Received_Normed', 'After_Money_Sent_Normed',
                      'After_Cashin_Normed', 'After_CashOut_Normed', 'After_UtilityBill_Normed']

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
other_before_features = ['Visited_Dar_Before_Moved', 'Rural_Ward']

# select features all:
select_features_all =  before_call_features + after_call_features + before_mfs_features + after_mfs_features +\
                       other_overall_features + other_before_features + region_features + dummy_region_features
# select features before:
select_features_before = region_features + before_call_features + before_mfs_features + other_before_features + \
                         region_features + dummy_region_features + ['H_Dist_to_Dar_Before']
# select features after:
select_features_after = after_call_features + after_mfs_features + ['Visited_University']

# save dfs:
df_features_selected = X_all[select_features_all]
df_features_selected.to_csv(X_data_path+"Selected_Features_Full_Reduced.csv")

df_before_features_selected = X_all[select_features_before]
df_before_features_selected.to_csv(X_data_path+"Selected_Features_Before_Reduced.csv")

df_after_features_selected = X_all[select_features_after]
df_after_features_selected.to_csv(X_data_path+"Selected_Features_After_Reduced.csv")

# look at correlation matrix:
X_and_y = df_features_selected.merge(y, how='inner', on="ID")
cor = round(X_and_y.corr(), 2)
cor_save_path = "Results/Descriptives/Correlations/"
cor.to_csv(cor_save_path+"y_and_X_all_Reduced_Region.csv")

X_and_yc = df_features_selected.merge(y_c, how='inner', on="ID")
cor = round(X_and_yc.corr(), 2)
cor_save_path = "Results/Descriptives/Correlations/"
cor.to_csv(cor_save_path+"yc_and_X_all_Reduced_Region.csv")

# count nas as a perc of data:
percent_missing = df_features_selected.isnull().sum() * 100 / len(df_features_selected)
missing_value_df = pd.DataFrame({'column_name': df_features_selected.columns,
                                 'percent_missing': percent_missing})
missing_data_path = "Data/Modelling/Missing_Data/"
missing_value_df.to_csv(missing_data_path+"Missing_all_features_Region.csv")

# select only those variables that have atleast a small cor with continous dv:
cor_cutoff = 0.07
cors = pd.DataFrame(cor['rank']).reset_index()
cor_vars = list(cors['index'][(cors['rank']>= cor_cutoff) | (cors['rank']<= -cor_cutoff)])
cor_vars.remove('rank')
df_all_features_selected_bycor = X_all[cor_vars]
df_all_features_selected_bycor.to_csv(X_data_path+"Selected_Features_All_Reduced_Bycors.csv")

# look at correlation matrix again:
X_and_yc = pd.concat([y_c, df_all_features_selected_bycor], axis=1)
cor = round(X_and_yc.corr(), 2)
cor_save_path = "Results/Descriptives/Correlations/"
cor.to_csv(cor_save_path+"y_and_X_all_Reduced_Region_Bycor.csv")

remove due to multicollinearity still:
remove_multicollinearity = ["Region_Population_Vulnerable_To_Poverty", "Region_Population_In_Severe_Poverty", "Region_Delivered_Facility",
                            "Region_Population_5_14years", "Region_Immunization_DPT"]
df_all_features_selected_bycor.drop(remove_multicollinearity, axis=1, inplace=True)

# save features selected:
df_all_features_selected_bycor.to_csv(X_data_path+"Selected_Features_All_Reduced_By_IV_and_DVcors{}.csv".format(cor_cutoff))

# t-tests:
n_tests = df_all_features_selected_bycor.shape[1]
bon_p = round(bonferroni_thresh(alpha_fwe=0.05, n=n_tests),4)
save_path = "Results/Feature_stat_tests/t_test_plots/t_test_plots_after_feature_selection/"
pvals = []
vars = []

X_and_y_reduced = df_all_features_selected_bycor.merge(y, how='inner', on="ID")

for var in df_all_features_selected_bycor.columns:
    t_test(X_and_y_reduced, a=0, b=1, save_path=save_path, group_col="Binary_DV", dv=var, fig_size=(6, 6),
           test_type="independent", save_name=var+t, title=var.replace("_"," "), data_shape='long', box=False,
           bon_corr_pthresh=bon_p, ylab=" ", xlab=" ", title_size=12)
    tval, pval = st.ttest_ind(a=X_and_y_reduced[var][X_and_y_reduced["Binary_DV"] == 0], b=X_and_y_reduced[var][X_and_y_reduced["Binary_DV"] == 1],
                        axis=0, equal_var=True, nan_policy='omit')
    pvals.append(pval)
    vars.append(var)

# FDR:
save_path = "Results/Feature_stat_tests/"
for fdr in [0.1, 0.05, 0.025, 0.01]:
    fdr_results, new_pvals = sm.stats.multitest.fdrcorrection(pvals, alpha=fdr, method='indep', is_sorted=False)
    fdr_dict = {"Feature":vars, "FDR_Result": fdr_results, "Orig_p_Value": pvals, "Adjusted_p_Value": new_pvals}
    fdr_df = pd.DataFrame.from_dict(fdr_dict, orient="index").T
    fdr_df.to_csv(save_path+"FDR_t_test_Results_{}{}.csv".format(fdr, t))

# look at graphs for all features (out of interest):
n_tests = df_features_selected.shape[1]
bon_p = round(bonferroni_thresh(alpha_fwe=0.05, n=n_tests),4)
save_path = "Results/Feature_stat_tests/t_test_plots/"
pvals = []
vars = []
X_and_y = df_features_selected.merge(y, how='inner', on="ID")

for var in df_features_selected.columns:
    t_test(X_and_y, a=0, b=1, save_path=save_path, group_col="Binary_DV", dv=var, fig_size=(6, 6),
           test_type="independent", save_name=var+t, title=var.replace("_"," "), data_shape='long', box=False,
           bon_corr_pthresh=bon_p, ylab=" ", xlab=" ", title_size=12)
    tval, pval = st.ttest_ind(a=X_and_y[var][X_and_y["Binary_DV"] == 0], b=X_and_y[var][X_and_y["Binary_DV"] == 1],
                        axis=0, equal_var=True, nan_policy='omit')
    pvals.append(pval)
    vars.append(var)


# t-tests - just for the after features:
n_tests = df_after_features_selected.shape[1]
bon_p = round(bonferroni_thresh(alpha_fwe=0.05, n=n_tests),4)
save_path = "Results/Feature_stat_tests/t_test_plots/t_test_plots_after_feature_selectionAFTER_FEATURES/"
pvals = []
vars = []

X_and_y_after = df_after_features_selected.merge(y, how='inner', on="ID")

ylabs = ['Mean no. of calls made', 'Entropy of numbers called', 'Mean calling distance (km)',
       'Percentage (%) of calls back home', 'Mean no. of calls made whilst visiting home',
       'Mean balance (TZS)', 'Amount into account per day (TZS)',
       'Amount out of account per day (TZS)', "Amount of 'cash in' per day (TZS)",
       "Amount of 'cash out' per day (TZS)", "Amount of 'utility bill' payments per day (TZS)",
       'Visited university = 1, Not = 0']

for var, ylab in zip(df_after_features_selected.columns, ylabs):
    t_test(X_and_y_after, a=0, b=1, save_path=save_path, group_col="Binary_DV", dv=var, fig_size=(6, 6),
           test_type="independent", save_name=var+t, title=" ", data_shape='long', box=False,
           bon_corr_pthresh=None, ylab=ylab, xlab=" ", title_size=12, aspect =0.5, xticklabs=[" ", " "])
    tval, pval = st.ttest_ind(a=X_and_y_after[var][X_and_y_after["Binary_DV"] == 0], b=X_and_y_after[var][X_and_y_after["Binary_DV"] == 1],
                        axis=0, equal_var=True, nan_policy='omit')
    pvals.append(pval)
    vars.append(var)

# FDR:
for fdr in [0.1, 0.05, 0.025, 0.01]:
    fdr_results, new_pvals = sm.stats.multitest.fdrcorrection(pvals, alpha=fdr, method='indep', is_sorted=False)
    fdr_dict = {"Feature":vars, "FDR_Result": fdr_results, "Orig_p_Value": pvals, "Adjusted_p_Value": new_pvals}
    fdr_df = pd.DataFrame.from_dict(fdr_dict, orient="index").T
    fdr_df.to_csv(save_path+"FDR_t_test_Results_{}{}.csv".format(fdr, t))


# 2x2 anovas:

before_after_vars = ['Before_Money_Sent_Normed', 'After_Money_Sent_Normed']
anova_df = X_and_y[['Binary_DV'] + before_after_vars]
anova_df.reset_index(inplace=True)
dv_df = X_and_y["Binary_DV"].reset_index()
anova_df.rename(columns ={"Before_Money_Sent_Normed":"Before", "After_Money_Sent_Normed":"After"}, inplace=True)
anova_melt_df = pd.melt(anova_df, id_vars='ID', var_name='Before_After',value_name='Money_Out',
                        value_vars=['Before', 'After'])
full_anova_melt_df = dv_df.merge(anova_melt_df, on="ID").reset_index()

save_path="Results/Feature_stat_tests/ANOVA/"
Twoway_ANOVA(df=full_anova_melt_df, group='Binary_DV', x='Before_After', y="Money_Out")

Plot_ANOVA(df=full_anova_melt_df, group='Binary_DV', x='Before_After', y="Money_Out",
           save_path = save_path, save_name = "Money_out_2x2",
           fontsize=12, legendloc="upper right",
           xorder_asc=False, labs = ['Not Vulnerable', 'Vulnerable'],
           ylab="Money out per day (TZS)")

# ----- Money rec ---------------------------------------------
before_after_vars = ['Before_Money_Received_Normed', 'After_Money_Received_Normed']
anova_df = X_and_y[['Binary_DV'] + before_after_vars]
anova_df.reset_index(inplace=True)
dv_df = X_and_y["Binary_DV"].reset_index()
anova_df.rename(columns ={"Before_Money_Received_Normed":"Before", "After_Money_Received_Normed":"After"}, inplace=True)
anova_melt_df = pd.melt(anova_df, id_vars='ID', var_name='Before_After',value_name='Money_Out',
                        value_vars=['Before', 'After'])
full_anova_melt_df = dv_df.merge(anova_melt_df, on="ID").reset_index()

save_path="Results/Feature_stat_tests/ANOVA/"
Twoway_ANOVA(df=full_anova_melt_df, group='Binary_DV', x='Before_After', y="Money_Out")

Plot_ANOVA(df=full_anova_melt_df, group='Binary_DV', x='Before_After', y="Money_Out",
           save_path = save_path, save_name = "Money_in_2x2",
           fontsize=12, legendloc="upper left",
           xorder_asc=False, labs = ['Not Vulnerable', 'Vulnerable'],
           ylab="Money in (all types) per day (TZS)")

# ----- Cashin ------------------------------------------------
before_after_vars = ['Before_Cashin_Normed', 'After_Cashin_Normed']
anova_df = X_and_y[['Binary_DV'] + before_after_vars]
anova_df.reset_index(inplace=True)
dv_df = X_and_y["Binary_DV"].reset_index()
anova_df.rename(columns ={"Before_Cashin_Normed":"Before", "After_Cashin_Normed":"After"}, inplace=True)
anova_melt_df = pd.melt(anova_df, id_vars='ID', var_name='Before_After',value_name='Money_Out',
                        value_vars=['Before', 'After'])
full_anova_melt_df = dv_df.merge(anova_melt_df, on="ID").reset_index()

save_path="Results/Feature_stat_tests/ANOVA/"
Twoway_ANOVA(df=full_anova_melt_df, group='Binary_DV', x='Before_After', y="Money_Out")

Plot_ANOVA(df=full_anova_melt_df, group='Binary_DV', x='Before_After', y="Money_Out",
           save_path = save_path, save_name = "Cash_in_2x2",
           fontsize=12, legendloc="upper left",
           xorder_asc=False, labs = ['Not Vulnerable', 'Vulnerable'],
           ylab="Cash in per day (TZS)")




print('done!')