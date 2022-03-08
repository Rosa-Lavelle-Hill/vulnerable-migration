import pandas as pd
import os
import numpy as np
import scipy.stats as st
from scipy.stats import chi2_contingency
from Functions.Database import select
from Functions.Plotting import plot_hist, plot_hist_stacked, PlotOverallGroup
from Functions.Preprocessing import fill_empty_groups, wide_to_long
from Functions.Stat_tests import t_test, bonferroni_thresh, PlotChiSquare


dir = "Data/Completed_Labels/"

dfs = []

for filename in os.listdir(dir):
    if filename.endswith(".csv"):
        with open(os.path.join(dir, filename), 'r') as f:
            print(filename)
            df = pd.read_csv(f)
            dfs.append(df)

df_labels = pd.concat(dfs, axis=0)


f = open("SQL/Queries/Select/Moved_to_Dar_final_Relaxed.txt")
df_q = f.read()

if __name__ == '__main__':
    df = select(df_q)

# clean:
df_labels.drop('id_num', inplace=True, axis=1)
df_labels.dropna(how='all', axis=0, inplace=True)

df['move_date'] =  df['move_dar_date'].dt.date
df.drop(columns =['move_dar_date','end_dar_date'], inplace=True)

# join:
df = df.merge(df_labels, on='customer_id', how='outer')

# add before/after col:
df['Group'] = 'na'
df['Group'][df['transfer_date']<df['move_date']] = 'Before'
df['Group'][df['transfer_date']>=df['move_date']] = 'After'

# count labels
print(df_labels['Moved'].value_counts())
print(df_labels['University'].value_counts())

#  only use examples with correct label
df = df[df['Moved'] == 1]

# Only use sample where No_Temeke------------------------:

y = pd.read_csv("Data/Modelling/DV/Binary_Dependent_Variable_50_No_Temeke.csv",
                index_col=None, usecols=["ID", "Binary_DV"])
y.rename(columns={"ID":"customer_id"}, inplace=True)

# join
df = df.merge(y, how='inner', on="customer_id")

# run analysis...
# -----------------------------------------------------------------------
# Descriptives of whole sample:

save_path = 'Results/Moved_to_Dar_Labelled_No_Temeke/Descriptives/'

# Time in Dar:
group = df[['customer_id','length_full_period_days']].groupby(by='customer_id', as_index=False).first()

plot_hist(save_name='Period_length',
          x=group['length_full_period_days'],
          bins=20,
          save_path=save_path,
          title= "Length of Period (days) in Dar",
          xlim=None,
          fig_size=(6, 6))

# stacked (with days in dar over period):

group = df[['customer_id','length_full_period_days', 'days_in_dar_over_period']].groupby(by='customer_id', as_index=False).first()

plot_hist_stacked(save_name='Period_length_days_in_dar_stacked',
                  x1=group['length_full_period_days'],
                  x2=group['days_in_dar_over_period'],
                  bins=20,
                  save_path=save_path,
                  title= "Length of Period (days) in Dar",
                  xlim=None,
                  fig_size=(6, 6),
                  x1_lab="Full Period (days)",
                  x2_lab="Days in Dar over period (days)")

# Mean balance:
group = df[['customer_id','pre_balance']].groupby(by='customer_id', as_index=False).mean()
plot_hist(save_name='Mean Balance',
          x=group['pre_balance'],
          bins=20,
          save_path=save_path,
          title= "Mean Balance (TZS)",
          xlim=None,
          fig_size=(6, 6))

# -----------------------------------------------------------------------
# Before/after tests

save_path= 'Results/Moved_to_Dar_Labelled_No_Temeke/Before_after_Tests/'

# Control for number of t-tests running
num_test = 3
p_thresh = round(bonferroni_thresh(0.05, num_test), 5)
print("t-tests Bonferroni p-thresh: {}".format(p_thresh))

# 1. mean pre-balance:
mean_bal = df[['customer_id','Group','pre_balance']].groupby(by=['customer_id', 'Group']).mean().reset_index()


t_test(mean_bal, a= 'Before', b= 'After',
       test_type='paired',
       data_shape = 'long',
       save_path=save_path,
       group_col='Group',
       dv='pre_balance',
       fig_size=(8,8),
       save_name= 'Mean_Balance',
       title= 'Mean Balance',
       ylab= "Mean Balance (TZS)",
       bon_corr_pthresh=p_thresh,
       box=True)

PlotOverallGroup(df=mean_bal, x='Group', y='pre_balance',
                 hue='Group', save_path=save_path,
                 save_name= "Median_Balance",
                 figsize=(8,8), kind='bar', ci=None,
                 title="Median Balance",
                 ylab="Median Balance",
                 estimator=np.median)

# -------------------------------------------------------------
# 2. sum of amount received:

# standardise by days of before/after:

first_date = df[['customer_id', 'transfer_date']].groupby(
              by='customer_id', dropna=False).min().reset_index()
first_date.columns = ['customer_id', 'first_date']

# join to df
df = df.merge(first_date, how='left', on= "customer_id")
df['days_before_dar'] = (df['move_date'] - df['first_date']).dt.days

sum_trans = df[['customer_id', 'days_before_dar', 'length_full_period_days','sender_receiver',
                'Group', 'transaction_amount']].groupby(
              by=['customer_id', 'days_before_dar', 'length_full_period_days', 'Group', 'sender_receiver'], dropna=False).sum().reset_index()

sum_trans = sum_trans[sum_trans['days_before_dar']>0]

# create sum_trans dv normalised by days:
sum_trans['sum_trans_div_days'] = np.nan
sum_trans['sum_trans_div_days'][sum_trans['Group']=='Before'] = sum_trans['transaction_amount']/sum_trans['days_before_dar']
sum_trans['sum_trans_div_days'][sum_trans['Group']=='After'] = sum_trans['transaction_amount']/sum_trans['length_full_period_days']

# create data with empty groups:
sum_trans = fill_empty_groups(sum_trans)

# just those received:
sum_rec = sum_trans[sum_trans['sender_receiver']=='Receiver']

t_test(sum_rec, a= 'Before', b= 'After',
       test_type='paired',
       subset='days_before_dar',
       data_shape = 'long',
       save_path=save_path,
       group_col='Group',
       dv='sum_trans_div_days',
       fig_size=(8,8),
       save_name= 'Sum_Transactions_Received_Div_Days',
       title= 'Sum Money In / Days',
       ylab= "Sum money in (TZS) / days",
       bon_corr_pthresh=p_thresh,
       box=True)

# -------------------------------------------------------------
# 3. sum of amount sent / days:

# just those received:
sum_sent = sum_trans[sum_trans['sender_receiver']=='Sender']

t_test(sum_sent, a= 'Before', b= 'After',
       test_type='paired',
       subset='days_before_dar',
       data_shape = 'long',
       save_path=save_path,
       group_col='Group',
       dv='sum_trans_div_days',
       fig_size=(8,8),
       save_name= 'Sum_Transactions_Sent_Div_Days',
       title= 'Sum Money Out / Days',
       ylab= "Sum money out (TZS) / days",
       bon_corr_pthresh=p_thresh,
       box=True)

# ===============================================================
# group customers into those who become better off, and those who didn't:

# 1st based on mean balance before/after:

df_pivot = mean_bal.pivot(index='customer_id', columns='Group', values='pre_balance')
df_pivot.reset_index(inplace=True)
df_pivot.sort_values(by='customer_id')
df_pivot.dropna(inplace=True, axis=0)

df_sucess = df_pivot[df_pivot["After"] > df_pivot["Before"]]
df_failure = df_pivot[df_pivot["After"] <= df_pivot["Before"]]

df_pivot['Outcome'] = "Not Known"
df_pivot['Outcome'][df_pivot["After"] > df_pivot["Before"]] = "Success"
df_pivot['Outcome'][df_pivot["Before"] > df_pivot["After"]] = "Failure"

print(df_pivot['Outcome'].value_counts())

# make long df:

df_long = pd.melt(df_pivot, id_vars=["customer_id", "Outcome"],
        var_name="Before_After", value_name="Balance")
df_long.sort_values(by=["customer_id", "Outcome", "Before_After"], inplace=True)

save_path= 'Results/Moved_to_Dar_Labelled_No_Temeke/Sucess_Failure/'

PlotOverallGroup(df=df_long, x='Outcome', y='Balance',
                 hue='Before_After', save_path=save_path,
                 save_name= "Mean_Balance_2way",
                 figsize=(8,8), kind='bar', ci=None,
                 title="Mean Balance",
                 ylab="Mean Balance",
                 estimator=np.mean,
                 legend=True)

# such variance in balance... differences might not be signficant...
# ----------------------------------------------------------------------------
#paired t-test for balance before and balance after in 1) sucess group, and 2) failure group

df_sucess2 = df_long[df_long['Outcome']=='Success']
df_failure2 = df_long[df_long['Outcome']=='Failure']

# 1) Sucess before and after:
df_sucess2_Before = df_sucess2[df_sucess2['Before_After']=='Before']
df_sucess2_Before.sort_values(by='customer_id', inplace=True)
df_sucess2_After = df_sucess2[df_sucess2['Before_After']=='After']
df_sucess2_After.sort_values(by='customer_id', inplace=True)
print('Sucess')
t, p = st.ttest_rel(a=df_sucess2_Before['Balance'],
                    b=df_sucess2_After['Balance'], axis=0,
                                nan_policy='raise')
t_string, p_string = F'{t:.4f}', f'{p:.4f}'
if p < 0.05:
        print("t= " + t_string + ", p= " + p_string + " * ")
else:
    print("t= " + t_string + ", p= " + p_string + " (Not significant)")


# 2) Failure before and after:
df_failure2_Before = df_failure2[df_failure2['Before_After']=='Before']
df_failure2_Before.sort_values(by='customer_id', inplace=True)
df_failure2_After = df_failure2[df_failure2['Before_After']=='After']
df_failure2_After.sort_values(by='customer_id', inplace=True)
print('Failure')
t, p = st.ttest_rel(a=df_failure2_Before['Balance'],
                    b=df_failure2_After['Balance'], axis=0,
                                nan_policy='raise')
t_string, p_string = F'{t:.4f}', f'{p:.4f}'
if p < 0.05:
        print("t= " + t_string + ", p= " + p_string + " * ")
else:
    print("t= " + t_string + ", p= " + p_string + " (Not significant)")

# ----------------------------------------------------------------------------
# create high/low balance before groups:

df_before = df_long[df_long['Before_After']=='Before']
df_after = df_long[df_long['Before_After']=='After']
mean_bal_before_df = df_before[['customer_id','Balance']].groupby(by='customer_id').mean()
med_bal_before = mean_bal_before_df.median()

df_long['Median_Before_Bal']= list(np.repeat(med_bal_before, df_long.shape[0]))
df_long['Before_Balance_Group']=np.nan
df_long['Before_Balance_Group'][(df_long['Before_After']=='Before') & (df_long['Balance']>=df_long['Median_Before_Bal'])]='High'
df_long['Before_Balance_Group'][(df_long['Before_After']=='Before') & (df_long['Balance']<df_long['Median_Before_Bal'])]='Low'

# fill nas using customer number:
df_long["Before_Balance_Group"] = df_long.groupby(['customer_id'], sort=False)['Before_Balance_Group'].apply(lambda x: x.ffill().bfill())

# Does pre-balance before move predict success?
counts = df_long[['Before_Balance_Group','Outcome','customer_id']].drop_duplicates()
counts = counts.groupby(['Before_Balance_Group','Outcome']).count()
counts.reset_index(inplace=True)

PlotOverallGroup(df=counts, x='Before_Balance_Group', y='customer_id',
                 hue='Outcome', save_path=save_path,
                 save_name= "Count_groups_Before_bal",
                 figsize=(8,8), kind='bar', ci=None,
                 title="Number of Individuals",
                 ylab="Number of individuals",
                 estimator=np.mean,
                 legend=True,
                 xlab="Balance Group Before Moving")

# chi square test:
counts_wide = counts.pivot(index='Before_Balance_Group', columns='Outcome', values='customer_id')
res = pd.DataFrame(chi2_contingency(counts_wide))
res = res.transpose()
res.columns = ["Pearson's Chi-sq indep (X^2)", "p-value", "(df)", "Expected values"]
res.drop("Expected values", axis=1, inplace=True)
res["p-value"] = res["p-value"].astype(float).round(4)
res["(df)"] = "(" + str(res["(df)"][0]) + ")"
with pd.option_context('display.float_format', '{:0.4f}'.format):
    print(res.to_string(index=False))


# histogram of mean pre-balance

 # plot x=before bal group, y=after balance group
save_path = 'Results/Moved_to_Dar_Labelled_No_Temeke/Descriptives/'
plot_hist(save_name='Mean Balance Before',
          x=df_before['Balance'],
          bins=100,
          save_path=save_path,
          title= "Mean balance before moving (TZS)",
          fig_size=(6, 6),
          xlim=(0,200000),
          ylim=(0,150))
plot_hist(save_name='Mean Balance After',
          x=df_after['Balance'],
          bins=100,
          save_path=save_path,
          title= "Mean balance after moving (TZS)",
          xlim=(0,200000),
          fig_size=(6, 6),
          ylim=(0,150))


#  Chi square test for different service types used (since moving):

# a) by Outcome
df_cat = df.merge(df_long[['customer_id','Outcome']], how='left', on='customer_id').drop_duplicates()

save_path= 'Results/Moved_to_Dar_Labelled_No_Temeke/Sucess_Failure/'

# Overall
PlotChiSquare(df=df_cat,
              x='Outcome',
              count_col='transfer_id',
              hue='service_type', save_path=save_path,
              save_name= "Count_service_types",
              figsize=(8,8), kind='bar', ci=None,
              title="Number of Transactions Overall",
              ylab="Number of Transactions",
              legend='out',
              aspect=1.5,
              xlab="Outcome Group",
              plot_counts_greater_than=700,
              after_only=False)

# After moving
PlotChiSquare(df=df_cat,
              x='Outcome',
              count_col='transfer_id',
              hue='service_type', save_path=save_path,
              save_name= "Count_service_types_After",
              figsize=(8,8), kind='bar', ci=None,
              title="Number of Transactions After moving to Dar",
              ylab="Number of Transactions",
              legend='out',
              aspect=1.5,
              xlab="Outcome Group",
              plot_counts_greater_than=700,
              after_only=True)


# After moving broken down by seder/rec
PlotChiSquare(df=df_cat,
              x='Outcome',
              count_col='transfer_id',
              hue='service_type', save_path=save_path,
              save_name= "Count_service_types_After_Send_rec",
              figsize=(8,8), kind='bar', ci=None,
              title="Number of Transactions After moving to Dar",
              ylab="Number of Transactions",
              legend='out',
              aspect=1.5,
              xlab="Outcome Group",
              plot_counts_greater_than=700,
              after_only=True,
              in_out_breakdown='sender_receiver')


# b) by before groups
df_cat = df.merge(df_long[['customer_id','Before_Balance_Group']], how='left', on='customer_id').drop_duplicates()

save_path= 'Results/Moved_to_Dar_Labelled_No_Temeke/Before_Bal_Groups/'

# Overall
PlotChiSquare(df=df_cat,
              x='Before_Balance_Group',
              count_col='transfer_id',
              hue='service_type', save_path=save_path,
              save_name= "Count_service_types",
              figsize=(8,8), kind='bar', ci=None,
              title="Number of Transactions Overall",
              ylab="Number of Transactions",
              legend='out',
              aspect=1.5,
              xlab="Before Balance Group",
              plot_counts_greater_than=700,
              after_only=False,
              bb= (1.21, 1))

# After only
PlotChiSquare(df=df_cat,
              x='Before_Balance_Group',
              count_col='transfer_id',
              hue='service_type', save_path=save_path,
              save_name= "Count_service_types_After",
              figsize=(8,8), kind='bar', ci=None,
              title="Number of Transactions After moving to Dar",
              ylab="Number of Transactions",
              legend='out',
              aspect=1.5,
              xlab="Before Balance Group",
              plot_counts_greater_than=700,
              after_only=True,
              bb= (1.21, 1))

# After broken down by sender/rec:
PlotChiSquare(df=df_cat,
              x='Before_Balance_Group',
              count_col='transfer_id',
              hue='service_type', save_path=save_path,
              save_name= "Count_service_types_After_Send_rec",
              figsize=(20,20), kind='bar', ci=None,
              title="Number of Transactions After moving to Dar",
              ylab="Number of Transactions",
              legend='out',
              aspect=1.5,
              xlab="Before Balance Group",
              plot_counts_greater_than=700,
              after_only=True,
              in_out_breakdown='sender_receiver',
              bb= (1.21, 1))


#  How do people's service type patterns change before/after moving?

df_cat = df.merge(df_long[['customer_id','Outcome']], how='left', on='customer_id').drop_duplicates()

save_path= 'Results/Moved_to_Dar_Labelled_No_Temeke/Before_after_Tests/'

# Standard groups
PlotChiSquare(df=df_cat,
              x='Group',
              count_col='transfer_id',
              hue='service_type', save_path=save_path,
              save_name= "Count_service_types_Normed",
              figsize=(8,8), kind='bar', ci=None,
              title="Number of Transactions Divided by Number of Days",
              ylab="Number of transactions divided by number of days",
              legend='out',
              aspect=1.5,
              xlab=" ",
              plot_counts_greater_than=0.09,
              after_only=False,
              bb= (1.01, 1),
              pallette="pastel",
              standardise={'Before':'days_before_dar',
                            'After': 'length_full_period_days'})

# Broken Down groups

PlotChiSquare(df=df_cat,
              x='Group',
              count_col='transfer_id',
              hue='service_type', save_path=save_path,
              save_name= "Count_service_types_Send_rec_Normed",
              figsize=(20,20), kind='bar', ci=None,
              title="Number of Transactions Divided by Number of Days",
              ylab="Number of transactions divided by number of days",
              legend='out',
              aspect=1.5,
              xlab=" ",
              plot_counts_greater_than=0.09,
              after_only=False,
              in_out_breakdown='sender_receiver',
              bb=(1.01, 1),
              pallette="pastel",
              standardise={'Before': 'days_before_dar',
                           'After': 'length_full_period_days'})

print('done!')

