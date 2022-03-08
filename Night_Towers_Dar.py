import datetime as dt
import pandas as pd
import os
import numpy as np

from Functions.Database import select
from Functions.Engineer_Features import calc_night_tower, after
from Functions.Preprocessing import drop_neg_labels
from Functions.Select_Feature_Data import select_feature_data
# --------------------------------------------------------------------------------------------------------
# Get labels
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

# --------------------------------------------------------------------------------------------------------
# Run ...
#  Create features on batches of the call data (data too large to load at once)
IDs_first= range(1,1401,100)
IDs_last= range(100,1400,100)
no_batches = 13

counter = 1
for first_id, last_id in zip(IDs_first, IDs_last):
    batch_start = dt.datetime.now()

    if __name__ == '__main__':
        df_calls = select_feature_data("SQL/Queries/Select/Features/Calls.txt", first_id, last_id)
        # Only keep those with positive label:
        df_calls = drop_neg_labels(df_calls, df_labels)
        IDs = df_calls['customer_id'].unique()

        # night tower
        df_calls_after = after(df_calls)
        night_tower_dar = calc_night_tower(X=df_calls_after, x_perc=10)
        if counter == 1:
            night_tower_df = pd.DataFrame.from_dict(night_tower_dar, orient='index')
        else:
            night_tower_df = night_tower_df.append(pd.DataFrame.from_dict(night_tower_dar, orient='index'))

        print('End of batch {}/{}'.format(counter, no_batches))
        batch_time = dt.datetime.now() - batch_start
        print('Batch run time: ' + str(batch_time))
        counter = counter + 1


# get night tower df:
save_path = "Data/Dar_Night_Towers/"
night_tower_df = night_tower_df.reset_index()
night_tower_df.columns = ["ID", "Dar_Night_Tower"]
night_tower_df.to_csv(save_path+"Dar_Night_Towers.csv")

# join


print('done')