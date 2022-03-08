import datetime as dt
import pandas as pd
import os
import numpy as np
import scipy
import scipy.stats as st
from Functions.Database import select
from Functions.Engineer_Features import HDistance, CountCalls, before, after, EntropyCalls, CallingDistance, \
    visit_university, ward_urban_or_rural, MobileMoney, calc_night_tower, calc_home_tower_outside_Dar, \
    calls_at_home, SumMoney
from Functions.Plotting import plot_hist, plot_hist_stacked
from Functions.Preprocessing import add_before_after_col, drop_neg_labels
from Functions.Select_Feature_Data import select_feature_data
from sklearn.preprocessing import OneHotEncoder

start_time = dt.datetime.now()

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
# ------------------------------------------------------------------------------------------------------
# run analysis...

#  Create features on batches of the call data (data too large to load at once)
# IDs_first= range(1,1401,100)
# IDs_last= range(100,1400,100)
# no_batches = 13

IDs_first= range(1,1251,50)
IDs_last= range(50,1300,50)
no_batches = 25

# # Test ranges:
# IDs_first= range(1101,1131,10)
# IDs_last= range(1110,1140,10)
# no_batches = 3

# IDs_first = [1]
# IDs_last = [3]

errdict={}
feature_dict = {}
counter = 1
plot_error = False

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

        # 1. Distance migrated:
        hdistance = HDistance(df_calls)

        # a) as the mean lat/lon not in Dar
        h_dist_mean = hdistance.d_mean()

        # b) as the mode (lat,lon) tuple not in Dar
        h_dist_mode = hdistance.d_mode()

        if plot_error == True:
            dist_err = hdistance.d_err(h_dist_mean, h_dist_mode, err_km=200, plot=True, save_path="Results/Descriptives/Features/Error_Distance_3D_Plots/")

        # c) dist from Dar (center point) for use as a before feature (when don't have info on where moving to)
        df_calls_before = before(df_calls)
        hdist_before = HDistance(df_calls_before)
        h_dist_to_Dar = hdist_before.dist_to_Dar(colname = 'H_Dist_to_Dar_Before')

        print('Distance features completed')
        # 2. Number of calls in/out (default is normed=True so actually counting calls per day):
        # a) overall
        count_call = CountCalls(df_calls)

        # i) all:
        overall_all_calls = count_call.count_calls(sent_or_rec='both', col_name="Overall_Sent_and_Received_Calls")
        # ii) received:
        overall_rec_calls = count_call.count_calls(sent_or_rec='received', col_name="Overall_Received_Calls")
        # iii) outward:
        overall_sent_calls = count_call.count_calls(sent_or_rec='sent', col_name='Overall_Sent_Calls')

        # b) before
        count_call_before = CountCalls(df_calls_before)

        # i) all
        before_all_calls = count_call_before.count_calls(sent_or_rec='both', col_name='Before_Sent_and_Received_Calls')
        # ii) received:
        before_rec_calls = count_call_before.count_calls(sent_or_rec='received', col_name='Before_Received_Calls')
        # iii) outward:
        before_sent_calls = count_call_before.count_calls(sent_or_rec='sent', col_name='Before_Sent_Calls')

        # c) after
        df_calls_after = after(df_calls)
        count_call_after = CountCalls(df_calls_after)

        # i) all
        after_all_calls = count_call_after.count_calls(sent_or_rec='both', col_name='After_Sent_and_Received_Calls')
        # ii) received:
        after_rec_calls = count_call_after.count_calls(sent_or_rec='received', col_name='After_Received_Calls')
        # iii) outward:
        after_sent_calls = count_call_after.count_calls(sent_or_rec='sent', col_name='After_Sent_Calls')
        print('Count call features completed')

        # 3. Entropy of Numbers called (received too affected by sampling bias)
        df_entropy = select_feature_data("SQL/Queries/Select/Features/Entropy.txt", first_id, last_id)
        df_entropy = drop_neg_labels(df_entropy, df_labels)
        # add move date:
        df_entropy = df_entropy.merge(df[['customer_id','move_date']].drop_duplicates(), on='customer_id', how='left')
        df_entropy = add_before_after_col(df_entropy)

        # a) overall:
        entropy = EntropyCalls(df_entropy)
        overall_entropy_calls = entropy.entropy_calls(colname="Overall_Call_Entropy")

        # b) before:
        df_entropy_before = before(df_entropy)
        entropy_before = EntropyCalls(df_entropy_before)
        before_entropy_calls = entropy_before.entropy_calls(colname="Before_Call_Entropy")

        # c) after:
        df_entropy_after = after(df_entropy)
        entropy_after = EntropyCalls(df_entropy_after)
        after_entropy_calls = entropy_after.entropy_calls(colname="After_Call_Entropy")
        print('Entropy call features completed')

        # 4. Calling Distance
        df_call_dist = select_feature_data("SQL/Queries/Select/Features/Calling_Distance.txt",first_id,last_id, id_rep=2)
        df_call_dist = drop_neg_labels(df_call_dist, df_labels)

        # add move date:
        df_call_dist = df_call_dist.merge(df[['customer_id', 'move_date']].drop_duplicates(), on='customer_id', how='left')
        df_call_dist = add_before_after_col(df_call_dist)

        # 4.a) Mean calling distance
        # i) overall
        overall_call_dist = CallingDistance(df_call_dist)
        overall_mean_call_distance = overall_call_dist.calc_mean_dist(colname="Overall_Mean_Calling_Distance")
        # ii) before
        df_call_dist_before = before(df_call_dist)
        before_call_dist = CallingDistance(df_call_dist_before)
        before_mean_call_distance = before_call_dist.calc_mean_dist(colname="Before_Mean_Calling_Distance")
        # iii) after
        df_call_dist_after = after(df_call_dist)
        after_call_dist = CallingDistance(df_call_dist_after)
        after_mean_call_distance = after_call_dist.calc_mean_dist(colname="After_Mean_Calling_Distance")

        # 4.b) Mode calling distance (rounded to nearest 20km)
        # i) overall
        overall_mode_call_distance = overall_call_dist.calc_mode_dist(colname="Overall_Mode_Calling_Distance")
        # ii) before
        before_mode_call_distance = before_call_dist.calc_mode_dist(colname="Before_Mode_Calling_Distance")
        # iii) after
        after_mode_call_distance = after_call_dist.calc_mode_dist(colname="After_Mode_Calling_Distance")

        # 4.c) Mean calling distance after moved divided by distance migrated (calc using mode)
        after_normed_call_distance = after_call_dist.calc_dist_normed(colname='After_Mean_Calling_Distance_Normed',
                                                                               migration_distance=h_dist_mode)
        # 4.d) Number of times called home tower (mode tower out of Dar before moved) after moving
        # calc most commonly called from tower (before moved, outside Dar)
        home_tower_df = calc_home_tower_outside_Dar(df_calls_before)
        # count calls to home tower after moved:
        after_perc_calls_home = after_call_dist.calc_number_of_home_calls(colname="After_Perc_Home_Calls",
                                                                               home_tower_df=home_tower_df)
        # 4.e) count visits home (calls from home tower) after moved:
        after_no_visits_home = calls_at_home(X= df_calls_after, colname="After_Number_of_Home_Visits",
                                             home_tower_df=home_tower_df)
        # 4.f) Perc of calls that were to Dar before moved:
        before_perc_calls_Dar = before_call_dist.calls_to_Dar(colname="Before_Perc_of_Calls_to_Dar")

        # 4.g) Where in Dar they called before moved (ward) rank:
        before_calls_Dar_ward_rank = before_call_dist.calls_to_Dar_ward(colname="Before_Calls_to_Dar_Ward_Rank",
                                                                        subwards=subwards)
        # 4.h) Where in Dar they called before moved (subard) rank:
        before_calls_Dar_subward_rank = before_call_dist.calls_to_Dar_subward(colname="Before_Calls_to_Dar_Subward_Rank",
                                                                          subwards=subwards, towers=towers)
        print('Mean calling distance features completed')
        # 5) Whether visited university campus or not:
        visited_university = visit_university(df=df_calls, label_df=df_labels, colname='Visited_University')

        # 6) First time to Dar?
        visited_dar_before_moved = count_call_before.visited_dar_before_moved(colname='Visited_Dar_Before_Moved')

        # 7) Where from?
        # a) what region from?
        region_from = count_call_before.where_from(colname="Region_From", level='region')
        # b) whether ward from is urban or rural
        df_urban_rural = select_feature_data("SQL/Queries/Select/Features/Urban_Rural_Wards.txt", first_id, last_id)
        ward_from = count_call_before.where_from(colname="Ward_From", level='wardcode')
        rural_ward = ward_urban_or_rural(colname= 'Rural_Ward', df_urban_rural=df_urban_rural, df_ward_from=ward_from)

        # 8) Mobile Money Features:
        df_mfs = df_mfs_full[df_mfs_full['customer_id'].isin(IDs)]
        # a) Mean balance

        # i) overall
        mfs_overall = MobileMoney(df_mfs)
        overall_mean_balance = mfs_overall.mean_balance(colname="Overall_Mean_Balance")

        # ii) before
        df_mfs_before = before(df_mfs, time_col='transfer_date')
        mfs_before = MobileMoney(df_mfs_before)
        before_mean_balance = mfs_before.mean_balance(colname="Before_Mean_Balance")

        # iii) after
        df_mfs_after = after(df_mfs, time_col='transfer_date')
        mfs_after = MobileMoney(df_mfs_after)
        after_mean_balance = mfs_after.mean_balance(colname="After_Mean_Balance")

        # b) Sum money in and out divided by days)
        # i) overall
        sum_money_overall = SumMoney(df_mfs)
        sum_money_overall_sent_rec_normed = sum_money_overall.overall(colname="Overall_Money_Sent_And_Rec_Normed", sender_rec="both")

        # ii) before
        sum_money_before = SumMoney(df_mfs_before)
        sum_money_before_sent_rec_normed = sum_money_before.before_after(colname="Before_Money_Sent_And_Rec_Normed",
                                                                    sender_rec="both", before_after='before')
        # iii) after
        sum_money_after = SumMoney(df_mfs_after)
        sum_money_after_sent_rec_normed = sum_money_after.before_after(colname="After_Money_Sent_And_Rec_Normed",
                                                                         sender_rec="both", before_after='after')
        # c) Sum money in (received) divided by days
        # i) overall
        sum_money_overall_rec_normed = sum_money_overall.overall(colname="Overall_Money_Received_Normed", sender_rec="Receiver")

        # ii) before
        sum_money_before_rec_normed = sum_money_before.before_after(colname="Before_Money_Received_Normed",
                                                                         sender_rec="Receiver", before_after='before')
        # iii) after
        sum_money_after_rec_normed = sum_money_after.before_after(colname="After_Money_Received_Normed",
                                                                    sender_rec="Receiver", before_after='after')
        # d) Sum money out (sent) divided by days
        # i) overall
        sum_money_overall_sent_normed = sum_money_overall.overall(colname="Overall_Money_Sent_Normed", sender_rec="Sender")

        # ii) before
        sum_money_before_sent_normed = sum_money_before.before_after(colname="Before_Money_Sent_Normed",
                                                                         sender_rec="Sender", before_after='before')
        # iii) after
        sum_money_after_sent_normed = sum_money_after.before_after(colname="After_Money_Sent_Normed",
                                                                    sender_rec="Sender", before_after='after')
        # e) Sum money sent and received in p2p transfers divided by days
        # i) overall
        sum_p2p_overall_sent_rec_normed = sum_money_overall.overall(colname="Overall_P2P_Sent_Rec_Normed", sender_rec="both",
                                                                service_type= "P2P transfer")
        # ii) before
        sum_p2p_before_sent_rec_normed = sum_money_before.before_after(colname="Before_P2P_Sent_Rec_Normed", before_after='before',
                                                                    sender_rec="both", service_type= "P2P transfer")
        # iii) after
        sum_p2p_after_sent_rec_normed = sum_money_after.before_after(colname="After_P2P_Sent_Rec_Normed", before_after='after',
                                                                    sender_rec="both", service_type= "P2P transfer")
        # f) Sum money sent in p2p transfers divided by days
        # i) overall
        sum_p2p_overall_sent_normed = sum_money_overall.overall(colname="Overall_P2P_Sent_Normed", sender_rec="Sender",
                                                                service_type= "P2P transfer")
        # ii) before
        sum_p2p_before_sent_normed = sum_money_before.before_after(colname="Before_P2P_Sent_Normed", before_after='before',
                                                                    sender_rec="Sender", service_type= "P2P transfer")
        # iii) after
        sum_p2p_after_sent_normed = sum_money_after.before_after(colname="After_P2P_Sent_Normed", before_after='after',
                                                                    sender_rec="Sender", service_type= "P2P transfer")
        # g) Sum money received in p2p transfers divided by days
        # i) overall
        sum_p2p_overall_rec_normed = sum_money_overall.overall(colname="Overall_P2P_Received_Normed", sender_rec="Receiver",
                                                                service_type= "P2P transfer")
        # ii) before
        sum_p2p_before_rec_normed = sum_money_before.before_after(colname="Before_P2P_Received_Normed", before_after='before',
                                                                    sender_rec="Receiver", service_type= "P2P transfer")
        # iii) after
        sum_p2p_after_rec_normed = sum_money_after.before_after(colname="After_P2P_Received_Normed", before_after='after',
                                                                    sender_rec="Receiver", service_type= "P2P transfer")
        # h) Sum "cash in" divided by days
        # i) overall
        sum_cashin_overall_normed = sum_money_overall.overall(colname="Overall_Cashin_Normed", sender_rec="both",
                                                                service_type= "Cash in")
        # ii) before
        sum_cashin_before_normed = sum_money_before.before_after(colname="Before_Cashin_Normed", sender_rec="both",
                                                                service_type= "Cash in", before_after="before")
        # iii) after
        sum_cashin_after_normed = sum_money_after.before_after(colname="After_Cashin_Normed", sender_rec="both",
                                                                service_type= "Cash in",  before_after="after")
        # i) Sum "cash Out" divided by days
        # i) overall
        sum_cashout_overall_normed = sum_money_overall.overall(colname="Overall_CashOut_Normed", sender_rec="both",
                                                                service_type= "Cash Out")
        # ii) before
        sum_cashout_before_normed = sum_money_before.before_after(colname="Before_CashOut_Normed", sender_rec="both",
                                                                service_type= "Cash Out", before_after="before")
        # iii) after
        sum_cashout_after_normed = sum_money_after.before_after(colname="After_CashOut_Normed", sender_rec="both",
                                                                service_type= "Cash Out",  before_after="after")
        # j) Sum "Utility BillPayment" divided by days
        # i) overall
        sum_utility_overall_normed = sum_money_overall.overall(colname="Overall_UtilityBill_Normed", sender_rec="both",
                                                                service_type= "Utility BillPayment")
        # ii) before
        sum_utility_before_normed = sum_money_before.before_after(colname="Before_UtilityBill_Normed", sender_rec="both",
                                                                service_type= "Utility BillPayment", before_after="before")
        # iii) after
        sum_utility_after_normed = sum_money_after.before_after(colname="After_UtilityBill_Normed", sender_rec="both",
                                                                service_type= "Utility BillPayment",  before_after="after")
        # k) Whether had mfs before moved or not:
        had_mfs_before = mfs_before.had_mfs_before(colname="Had_MFS_Before")

        # Add/append feature dfs to dictionary
        if counter == 1:
            feature_dict['Migration_Dist_Mean'] = h_dist_mean
            feature_dict['Migration_Dist_Mode'] = h_dist_mode
            feature_dict['H_Dist_to_Dar_Before'] = h_dist_to_Dar
            feature_dict['Overall_Sent_and_Received_Calls'] = overall_all_calls
            feature_dict['Overall_Received_Calls'] = overall_rec_calls
            feature_dict['Overall_Sent_Calls'] = overall_sent_calls
            feature_dict['Before_Sent_and_Received_Calls'] = before_all_calls
            feature_dict['Before_Received_Calls'] = before_rec_calls
            feature_dict['Before_Sent_Calls'] = before_sent_calls
            feature_dict['After_Sent_and_Received_Calls'] = after_all_calls
            feature_dict['After_Received_Calls'] = after_rec_calls
            feature_dict['After_Sent_Calls'] = after_sent_calls
            feature_dict['Overall_Entropy_Calls'] = overall_entropy_calls
            feature_dict['Before_Entropy_Calls'] = before_entropy_calls
            feature_dict['After_Entropy_Calls'] = after_entropy_calls
            feature_dict['Overall_Mean_Calling_Distance'] = overall_mean_call_distance
            feature_dict['Before_Mean_Calling_Distance'] = before_mean_call_distance
            feature_dict['After_Mean_Calling_Distance'] = after_mean_call_distance
            feature_dict['Overall_Mode_Calling_Distance'] = overall_mode_call_distance
            feature_dict['Before_Mode_Calling_Distance'] = before_mode_call_distance
            feature_dict['After_Mode_Calling_Distance'] = after_mode_call_distance
            feature_dict['After_Mean_Calling_Distance_Normed'] = after_normed_call_distance
            feature_dict['After_Perc_Home_Calls'] = after_perc_calls_home
            feature_dict['After_Number_of_Home_Visits'] = after_no_visits_home
            feature_dict['Before_Perc_of_Calls_to_Dar'] = before_perc_calls_Dar
            feature_dict['Visited_University'] = visited_university
            feature_dict['Visited_Dar_Before_Moved'] = visited_dar_before_moved
            feature_dict['Region_From'] = region_from
            feature_dict['Rural_Ward'] = rural_ward
            feature_dict['Overall_Mean_Balance'] = overall_mean_balance
            feature_dict['Before_Mean_Balance'] = before_mean_balance
            feature_dict['After_Mean_Balance'] = after_mean_balance
            feature_dict['Overall_Money_Sent_And_Rec_Normed'] = sum_money_overall_sent_rec_normed
            feature_dict['Before_Money_Sent_And_Rec_Normed'] = sum_money_before_sent_rec_normed
            feature_dict['After_Money_Sent_And_Rec_Normed'] = sum_money_after_sent_rec_normed
            feature_dict['Overall_Money_Received_Normed'] = sum_money_overall_rec_normed
            feature_dict['Before_Money_Received_Normed'] = sum_money_before_rec_normed
            feature_dict['After_Money_Received_Normed'] = sum_money_after_rec_normed
            feature_dict['Overall_Money_Sent_Normed'] = sum_money_overall_sent_normed
            feature_dict['Before_Money_Sent_Normed'] = sum_money_before_sent_normed
            feature_dict['After_Money_Sent_Normed'] = sum_money_after_sent_normed
            feature_dict['Overall_P2P_Sent_Rec_Normed'] = sum_p2p_overall_sent_rec_normed
            feature_dict['Before_P2P_Sent_Rec_Normed'] = sum_p2p_before_sent_rec_normed
            feature_dict['After_P2P_Sent_Rec_Normed'] = sum_p2p_after_sent_rec_normed
            feature_dict['Overall_P2P_Sent_Normed'] = sum_p2p_overall_sent_normed
            feature_dict['Before_P2P_Sent_Normed'] = sum_p2p_before_sent_normed
            feature_dict['After_P2P_Sent_Normed'] = sum_p2p_after_sent_normed
            feature_dict['Overall_P2P_Received_Normed'] = sum_p2p_overall_rec_normed
            feature_dict['Before_P2P_Received_Normed'] = sum_p2p_before_rec_normed
            feature_dict['After_P2P_Received_Normed'] = sum_p2p_after_rec_normed
            feature_dict['Overall_Cashin_Normed'] = sum_cashin_overall_normed
            feature_dict['Before_Cashin_Normed'] = sum_cashin_before_normed
            feature_dict['After_Cashin_Normed'] = sum_cashin_after_normed
            feature_dict['Overall_CashOut_Normed'] = sum_cashout_overall_normed
            feature_dict['Before_CashOut_Normed'] = sum_cashout_before_normed
            feature_dict['After_CashOut_Normed'] = sum_cashout_after_normed
            feature_dict['Overall_UtilityBill_Normed'] = sum_utility_overall_normed
            feature_dict['Before_UtilityBill_Normed'] = sum_utility_before_normed
            feature_dict['After_UtilityBill_Normed'] = sum_utility_after_normed
            feature_dict['Before_Calls_to_Dar_Ward_Rank'] = before_calls_Dar_ward_rank
            feature_dict['Before_Calls_to_Dar_Subward_Rank'] = before_calls_Dar_subward_rank
            feature_dict['Had_MFS_Before'] = had_mfs_before

        else:
            feature_dict['Migration_Dist_Mean'] = feature_dict['Migration_Dist_Mean'].append(h_dist_mean, ignore_index=True)
            feature_dict['Migration_Dist_Mode'] = feature_dict['Migration_Dist_Mode'].append(h_dist_mode, ignore_index=True)
            feature_dict['H_Dist_to_Dar_Before'] = feature_dict['H_Dist_to_Dar_Before'].append(h_dist_to_Dar, ignore_index=True)
            feature_dict['Overall_Sent_and_Received_Calls'] = feature_dict['Overall_Sent_and_Received_Calls'].append(overall_all_calls, ignore_index=True)
            feature_dict['Overall_Received_Calls'] = feature_dict['Overall_Received_Calls'].append(overall_rec_calls, ignore_index=True)
            feature_dict['Overall_Sent_Calls'] = feature_dict['Overall_Sent_Calls'].append(overall_sent_calls, ignore_index=True)
            feature_dict['Before_Sent_and_Received_Calls'] = feature_dict['Before_Sent_and_Received_Calls'].append(before_all_calls, ignore_index=True)
            feature_dict['Before_Received_Calls'] = feature_dict['Before_Received_Calls'].append(before_rec_calls, ignore_index=True)
            feature_dict['Before_Sent_Calls'] = feature_dict['Before_Sent_Calls'].append(before_sent_calls, ignore_index=True)
            feature_dict['After_Sent_and_Received_Calls'] = feature_dict['After_Sent_and_Received_Calls'].append(after_all_calls, ignore_index=True)
            feature_dict['After_Received_Calls'] = feature_dict['After_Received_Calls'].append(after_rec_calls, ignore_index=True)
            feature_dict['After_Sent_Calls'] = feature_dict['After_Sent_Calls'].append(after_sent_calls, ignore_index=True)
            feature_dict['Overall_Entropy_Calls'] = feature_dict['Overall_Entropy_Calls'].append(overall_entropy_calls, ignore_index=True)
            feature_dict['Before_Entropy_Calls'] = feature_dict['Before_Entropy_Calls'].append(before_entropy_calls, ignore_index=True)
            feature_dict['After_Entropy_Calls'] = feature_dict['After_Entropy_Calls'].append(after_entropy_calls,ignore_index=True)
            feature_dict['Overall_Mean_Calling_Distance'] = feature_dict['Overall_Mean_Calling_Distance'].append(overall_mean_call_distance,ignore_index=True)
            feature_dict['Before_Mean_Calling_Distance'] = feature_dict['Before_Mean_Calling_Distance'].append(before_mean_call_distance,ignore_index=True)
            feature_dict['After_Mean_Calling_Distance'] = feature_dict['After_Mean_Calling_Distance'].append(after_mean_call_distance,ignore_index=True)
            feature_dict['Overall_Mode_Calling_Distance'] = feature_dict['Overall_Mode_Calling_Distance'].append(overall_mode_call_distance,ignore_index=True)
            feature_dict['Before_Mode_Calling_Distance'] = feature_dict['Before_Mode_Calling_Distance'].append(before_mode_call_distance,ignore_index=True)
            feature_dict['After_Mode_Calling_Distance'] = feature_dict['After_Mode_Calling_Distance'].append(after_mode_call_distance,ignore_index=True)
            feature_dict['After_Mean_Calling_Distance_Normed'] = feature_dict['After_Mean_Calling_Distance_Normed'].append(after_normed_call_distance,ignore_index=True)
            feature_dict['After_Number_of_Home_Visits'] = feature_dict['After_Number_of_Home_Visits'].append(after_no_visits_home, ignore_index=True)
            feature_dict['After_Perc_Home_Calls'] = feature_dict['After_Perc_Home_Calls'].append(after_perc_calls_home, ignore_index=True)
            feature_dict['Before_Perc_of_Calls_to_Dar'] = feature_dict['Before_Perc_of_Calls_to_Dar'].append(before_perc_calls_Dar, ignore_index=True)
            feature_dict['Visited_University'] = feature_dict['Visited_University'].append(visited_university, ignore_index=True)
            feature_dict['Visited_Dar_Before_Moved'] = feature_dict['Visited_Dar_Before_Moved'].append(visited_dar_before_moved, ignore_index=True)
            feature_dict['Region_From'] = feature_dict['Region_From'].append(region_from, ignore_index=True)
            feature_dict['Rural_Ward'] = feature_dict['Rural_Ward'].append(rural_ward, ignore_index=True)
            feature_dict['Overall_Mean_Balance'] = feature_dict['Overall_Mean_Balance'].append(overall_mean_balance, ignore_index=True)
            feature_dict['Before_Mean_Balance'] = feature_dict['Before_Mean_Balance'].append(before_mean_balance, ignore_index=True)
            feature_dict['After_Mean_Balance'] = feature_dict['After_Mean_Balance'].append(after_mean_balance, ignore_index=True)
            feature_dict['Overall_Money_Sent_And_Rec_Normed'] = feature_dict['Overall_Money_Sent_And_Rec_Normed'].append(sum_money_overall_sent_rec_normed, ignore_index=True)
            feature_dict['Before_Money_Sent_And_Rec_Normed'] = feature_dict['Before_Money_Sent_And_Rec_Normed'].append(sum_money_before_sent_rec_normed, ignore_index=True)
            feature_dict['After_Money_Sent_And_Rec_Normed'] = feature_dict['After_Money_Sent_And_Rec_Normed'].append(sum_money_after_sent_rec_normed, ignore_index=True)
            feature_dict['Overall_Money_Received_Normed'] = feature_dict['Overall_Money_Received_Normed'].append(sum_money_overall_rec_normed, ignore_index=True)
            feature_dict['Before_Money_Received_Normed'] = feature_dict['Before_Money_Received_Normed'].append(sum_money_before_rec_normed, ignore_index=True)
            feature_dict['After_Money_Received_Normed'] = feature_dict['After_Money_Received_Normed'].append(sum_money_after_rec_normed, ignore_index=True)
            feature_dict['Overall_Money_Sent_Normed'] = feature_dict['Overall_Money_Sent_Normed'].append(sum_money_overall_sent_normed, ignore_index=True)
            feature_dict['Before_Money_Sent_Normed'] = feature_dict['Before_Money_Sent_Normed'].append(sum_money_before_sent_normed, ignore_index=True)
            feature_dict['After_Money_Sent_Normed'] = feature_dict['After_Money_Sent_Normed'].append(sum_money_after_sent_normed, ignore_index=True)
            feature_dict['Overall_P2P_Sent_Rec_Normed'] = feature_dict['Overall_P2P_Sent_Rec_Normed'].append(sum_p2p_overall_sent_rec_normed, ignore_index=True)
            feature_dict['Before_P2P_Sent_Rec_Normed'] = feature_dict['Before_P2P_Sent_Rec_Normed'].append(sum_p2p_before_sent_rec_normed, ignore_index=True)
            feature_dict['After_P2P_Sent_Rec_Normed'] = feature_dict['After_P2P_Sent_Rec_Normed'].append(sum_p2p_after_sent_rec_normed, ignore_index=True)
            feature_dict['Overall_P2P_Sent_Normed'] = feature_dict['Overall_P2P_Sent_Normed'].append(sum_p2p_overall_sent_normed, ignore_index=True)
            feature_dict['Before_P2P_Sent_Normed'] = feature_dict['Before_P2P_Sent_Normed'].append(sum_p2p_before_sent_normed, ignore_index=True)
            feature_dict['After_P2P_Sent_Normed'] = feature_dict['After_P2P_Sent_Normed'].append(sum_p2p_after_sent_normed, ignore_index=True)
            feature_dict['Overall_P2P_Received_Normed'] = feature_dict['Overall_P2P_Received_Normed'].append(sum_p2p_overall_rec_normed, ignore_index=True)
            feature_dict['Before_P2P_Received_Normed'] = feature_dict['Before_P2P_Received_Normed'].append(sum_p2p_before_rec_normed, ignore_index=True)
            feature_dict['After_P2P_Received_Normed'] = feature_dict['After_P2P_Received_Normed'].append(sum_p2p_after_rec_normed, ignore_index=True)
            feature_dict['Overall_Cashin_Normed'] = feature_dict['Overall_Cashin_Normed'].append(sum_cashin_overall_normed, ignore_index=True)
            feature_dict['Before_Cashin_Normed'] = feature_dict['Before_Cashin_Normed'].append(sum_cashin_before_normed, ignore_index=True)
            feature_dict['After_Cashin_Normed'] = feature_dict['After_Cashin_Normed'].append(sum_cashin_after_normed, ignore_index=True)
            feature_dict['Overall_CashOut_Normed'] = feature_dict['Overall_CashOut_Normed'].append(sum_cashout_overall_normed, ignore_index=True)
            feature_dict['Before_CashOut_Normed'] = feature_dict['Before_CashOut_Normed'].append(sum_cashout_before_normed, ignore_index=True)
            feature_dict['After_CashOut_Normed'] = feature_dict['After_CashOut_Normed'].append(sum_cashout_after_normed, ignore_index=True)
            feature_dict['Overall_UtilityBill_Normed'] = feature_dict['Overall_UtilityBill_Normed'].append(sum_utility_overall_normed, ignore_index=True)
            feature_dict['Before_UtilityBill_Normed'] = feature_dict['Before_UtilityBill_Normed'].append(sum_utility_before_normed, ignore_index=True)
            feature_dict['After_UtilityBill_Normed'] = feature_dict['After_UtilityBill_Normed'].append(sum_utility_after_normed, ignore_index=True)
            feature_dict['Before_Calls_to_Dar_Ward_Rank'] = feature_dict['Before_Calls_to_Dar_Ward_Rank'].append(before_calls_Dar_ward_rank, ignore_index=True)
            feature_dict['Before_Calls_to_Dar_Subward_Rank'] = feature_dict['Before_Calls_to_Dar_Subward_Rank'].append(before_calls_Dar_subward_rank, ignore_index=True)
            feature_dict['Had_MFS_Before'] = feature_dict['Had_MFS_Before'].append(had_mfs_before, ignore_index=True)

        print('End of batch {}/{}'.format(counter, no_batches))
        batch_time = dt.datetime.now() - batch_start
        print('Batch run time: ' + str(batch_time))
        counter = counter+1

# dict to dataframe:
for key, value in feature_dict.items():
    value.set_index('ID', inplace=True)
    feature_dict[key] = feature_dict[key].loc[~feature_dict[key].index.duplicated(keep='first')]

df = pd.concat(feature_dict.values(), axis=1, ignore_index=False)

# save
data_save_path = "Data/Modelling/Features/"
df.to_csv(data_save_path + "All_features_no_dummy_nas.csv")

runtime = dt.datetime.now() - start_time
print('Total run time: ' + str(runtime))

print('done!')