import pandas as pd
import os
import numpy as np

from Functions.Database import select
from Functions.Engineer_Features import before, after, CountCalls, MobileMoney
from Functions.Preprocessing import add_before_after_col, drop_neg_labels
from Functions.Select_Feature_Data import select_feature_data

dir = "Data/Completed_Labels/"

dfs = []

for filename in os.listdir(dir):
    if filename.endswith(".csv"):
        with open(os.path.join(dir, filename), 'r') as f:
            print(filename)
            df = pd.read_csv(f)
            dfs.append(df)

df_labels = pd.concat(dfs, axis=0)

# clean:
df_labels.drop('id_num', inplace=True, axis=1)
df_labels.dropna(how='all', axis=0, inplace=True)

f = open("SQL/Queries/Select/Moved_to_Dar_final_Relaxed.txt")
df_q = f.read()

if __name__ == '__main__':
    df = select(df_q)

df['move_date'] = df['move_dar_date'].dt.date
df.drop(columns =['move_dar_date','end_dar_date'], inplace=True)

# join:
df = df.merge(df_labels, on='customer_id', how='outer')

# add before/after col:
add_before_after_col(df, date_col='transfer_date',
                     move_date_col='move_date',
                     new_col_name="Before_After")


#  only use examples with correct label
df_mfs_full = df[df['Moved'] == 1]

# subwards ranking:
subwards = pd.read_csv("Data/Subwards/subward_quality.csv")
# subwards and towers:
towers = pd.read_csv("Data/Subwards/towers_subwards.csv")

IDs_first= range(1,1251,50)
IDs_last= range(50,1300,50)
no_batches = 25

errdict={}
feature_dict = {}
counter = 1
plot_error = False

for first_id, last_id in zip(IDs_first, IDs_last):

    if __name__ == '__main__':
        df_calls = select_feature_data("SQL/Queries/Select/Features/Calls.txt", first_id, last_id)
        # Only keep those with positive label:
        df_calls = drop_neg_labels(df_calls, df_labels)
        IDs = df_calls['customer_id'].unique()

        # count calls before
        df_calls_before = before(df_calls)
        count_call_before = CountCalls(df_calls_before)
        before_all_calls = count_call_before.count_calls(sent_or_rec='both', col_name='Before_all_Calls', normed=False)

        # count calls after
        df_calls_after = after(df_calls)
        count_call_after = CountCalls(df_calls_after)
        after_all_calls = count_call_after.count_calls(sent_or_rec='both', col_name='After_all_Calls', normed=False)

        # count mfs transactions before
        df_mfs = df_mfs_full[df_mfs_full['customer_id'].isin(IDs)]

        df_mfs_before = before(df_mfs, time_col='transfer_date')
        mfs_before = MobileMoney(df_mfs_before)
        before_count_trans = mfs_before.count_transactions(colname="Before_count_Trans")
        # count mfs transactions after

        df_mfs_after = after(df_mfs, time_col='transfer_date')
        mfs_after = MobileMoney(df_mfs_after)
        after_count_trans = mfs_after.count_transactions(colname="After_count_Trans")

        if counter == 1:
            feature_dict['Before_all_Calls'] = before_all_calls
            feature_dict['After_all_Calls'] = after_all_calls
            feature_dict['Before_count_Trans'] = before_count_trans
            feature_dict['After_count_Trans'] = after_count_trans

        else:
            feature_dict['Before_all_Calls'] = feature_dict['Before_all_Calls'].append(before_all_calls, ignore_index=True)
            feature_dict['After_all_Calls'] = feature_dict['After_all_Calls'].append(after_all_calls, ignore_index=True)
            feature_dict['Before_count_Trans'] = feature_dict['Before_count_Trans'].append(before_count_trans,ignore_index=True)
            feature_dict['After_count_Trans'] = feature_dict['After_count_Trans'].append(after_count_trans, ignore_index=True)

        counter = counter + 1

for key, value in feature_dict.items():
    value.set_index('ID', inplace=True)
    feature_dict[key] = feature_dict[key].loc[~feature_dict[key].index.duplicated(keep='first')]

df = pd.concat(feature_dict.values(), axis=1, ignore_index=False)

# descriptives:
print("Before calls - Mean:{}, Median:{}, SD:{}".format(round(df['Before_all_Calls'].mean(), 2),
                                                        round(df['Before_all_Calls'].median(), 2),
                                                        round(df['Before_all_Calls'].std(), 2)))

print("After calls - Mean:{}, Median:{}, SD:{}".format(round(df['After_all_Calls'].mean(), 2),
                                                        round(df['After_all_Calls'].median(), 2),
                                                        round(df['After_all_Calls'].std(), 2)))

print("Before transactions - Mean:{}, Median:{}, SD:{}".format(round(df['Before_count_Trans'].mean(), 2),
                                                        round(df['Before_count_Trans'].median(), 2),
                                                        round(df['Before_count_Trans'].std(), 2)))

print("After transactions - Mean:{}, Median:{}, SD:{}".format(round(df['After_count_Trans'].mean(), 2),
                                                        round(df['After_count_Trans'].median(), 2),
                                                        round(df['After_count_Trans'].std(), 2)))


print('done!')













