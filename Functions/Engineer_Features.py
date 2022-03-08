import pandas as pd
import os
import numpy as np
import scipy
import scipy.stats as st
from scipy.stats import chi2_contingency
from scipy.stats import mode
from Functions.Database import select
from Functions.Plotting import plot_hist, plot_hist_stacked, PlotOverallGroup, Plot_3D
from Functions.Preprocessing import fill_empty_groups, wide_to_long, add_before_after_col, fill_empty_groups_overall
from Functions.Stat_tests import t_test, bonferroni_thresh, PlotChiSquare
from math import sin, cos, sqrt, atan2, radians
from pyitlib import discrete_random_variable as drv
from scipy.stats import entropy


class HDistance(object):
    def __init__(self, X,
                 R = 6373.0):
        self.X = X
        self.R = R
        self.not_Dar = self.X[self.X['tower_region'] != 'Dar-es-salaam']
        self.Dar = self.X[self.X['tower_region'] == 'Dar-es-salaam']
        self.IDs = pd.DataFrame(self.X['customer_id'].unique(), columns=['IDs'])

    def dist_to_Dar(self, colname):
        """Calculates distance to center of Dar from mean lat lon from calls.
        Use on before data for feature.
        Lat and lon for Dar are from LatLon.net"""

        DlatY = pd.Series(np.repeat(radians(-6.776012), self.IDs.shape[0]))
        DlonX = pd.Series(np.repeat(radians(39.178326), self.IDs.shape[0]))

        mean_loc_not_Dar = self.not_Dar[['customer_id', 'tower_geom_x', 'tower_geom_y']].groupby('customer_id').mean().reset_index()
        # todo: use night tower calc instead of mean
        lon_Not = mean_loc_not_Dar['tower_geom_x'].map(radians)
        lat_Not = mean_loc_not_Dar['tower_geom_y'].map(radians)

        self.dist_to_Dar = self.get_dists(DlatY, lat_Not, DlonX, lon_Not)

        # calculate Haversine distance:
        hdist = self.haversine(self.dist_to_Dar, colname=colname)
        return hdist

    def d_mean(self):
        mean_loc_not_Dar = self.not_Dar[['customer_id', 'tower_geom_x', 'tower_geom_y']].groupby('customer_id').mean().reset_index()
        mean_loc_in_Dar = self.Dar[['customer_id', 'tower_geom_x', 'tower_geom_y']].groupby('customer_id').mean().reset_index()

        lon_Not = mean_loc_not_Dar['tower_geom_x'].map(radians)
        lat_Not = mean_loc_not_Dar['tower_geom_y'].map(radians)
        lon_Dar = mean_loc_in_Dar['tower_geom_x'].map(radians)
        lat_Dar = mean_loc_in_Dar['tower_geom_y'].map(radians)

        self.mean_dists = self.get_dists(lat_Dar, lat_Not, lon_Dar, lon_Not)

        # calculate Haversine distance:
        hdist_mean = self.haversine(self.mean_dists, colname='H_dist_Mean')
        return hdist_mean


    def d_err(self, d_hist_mean, d_hist_mode, err_km, save_path=None, plot=False):
        """Calculates the error between two distance measures and if plot==True
        plots examples where the error is greater than err_km"""

        Dist_Dict = {}
        err_df = d_hist_mean.merge(d_hist_mode, how='inner', on="ID")
        err_df["Error"] = abs(err_df['H_dist_Mean'] - err_df['H_dist_Mode'])
        for i, row in err_df.iterrows():
            if row["Error"] > err_km:
                Dist_Dict['Mean'] = self.mean_dists[self.mean_dists['ID'] == row['ID']]
                Dist_Dict['Mode'] = self.mode_dists[self.mode_dists['ID'] == row['ID']]
                #plot:
                if plot==True:
                    id_X = self.X[self.X['customer_id'] == row['ID']]
                    Plot_3D(df=id_X, save_path=save_path,
                            x='tower_geom_x',
                            y='tower_geom_y',
                            move_date_col='move_date',
                            region_col='tower_region',
                            type='scatter',
                            uni_strict=True,
                            n_graphs=20,
                            c_uni=None,
                            time_col='timestamp',
                            id_col='customer_id',
                            c_dar='m',
                            c_not='c',
                            id_count_col='id_num',
                            Dist_Dict=Dist_Dict,
                            show=False
                            )


    def d_mode(self):

        self.not_Dar['Tuple'] = list(zip(self.not_Dar['tower_geom_x'], self.not_Dar['tower_geom_y']))
        self.Dar['Tuple'] = list(zip(self.Dar['tower_geom_x'], self.Dar['tower_geom_y']))

        # Not Dar
        Not_Lon = self.not_Dar[['customer_id', 'Tuple']].groupby('customer_id')['Tuple'].apply(
            lambda x: x.mode()[0][0]).reset_index()
        Not_Lon.columns = ['ID', 'Lon']
        lon_Not = Not_Lon['Lon'].map(radians)
        Not_Lat = self.not_Dar[['customer_id', 'Tuple']].groupby('customer_id')['Tuple'].apply(
            lambda x: x.mode()[0][1]).reset_index()
        Not_Lat.columns = ['ID', 'Lat']
        lat_Not = Not_Lat['Lat'].map(radians)

        # Dar
        Dar_Lon = self.Dar[['customer_id', 'Tuple']].groupby('customer_id')['Tuple'].apply(
            lambda x: x.mode()[0][0]).reset_index()
        Dar_Lon.columns = ['ID', 'Lon']
        lon_Dar = Dar_Lon['Lon'].map(radians)
        Dar_Lat = self.Dar[['customer_id', 'Tuple']].groupby('customer_id')['Tuple'].apply(
            lambda x: x.mode()[0][1]).reset_index()
        Dar_Lat.columns = ['ID', 'Lat']
        lat_Dar = Dar_Lat['Lat'].map(radians)

        self.mode_dists = self.get_dists(lat_Dar, lat_Not, lon_Dar, lon_Not)

        # calculate Haversine distance:
        hdist_mode = self.haversine(self.mode_dists, colname='H_dist_Mode')
        return hdist_mode


    def haversine(self, dists, colname):
        hdist_dict = {}
        for i, row in dists.iterrows():
            a = sin(row.Lat_dist / 2) ** 2 + cos(row.Lat_Not) * cos(row.Lat_Dar) * sin(row.Lon_dist / 2) ** 2
            c = 2 * atan2(sqrt(a), sqrt(1 - a))
            Haversine_dist_kms = self.R * c
            hdist_dict[row['ID']] = Haversine_dist_kms
        hdist_mean = pd.DataFrame.from_dict(hdist_dict, orient='index').reset_index()
        hdist_mean.columns = ['ID', colname]
        return hdist_mean


    def get_dists(self, lat_Dar, lat_Not, lon_Dar, lon_Not):
        dlon = lon_Dar - lon_Not
        dlat = lat_Dar - lat_Not
        series = [self.IDs, dlat, dlon, lat_Dar, lon_Dar, lat_Not, lon_Not]
        dists = pd.concat(series, axis=1)
        dists.columns = ['ID', 'Lat_dist', 'Lon_dist', 'Lat_Dar', 'Lon_Dar', 'Lat_Not', 'Lon_Not']
        return dists

def calc_home_tower_outside_Dar(X):
    not_Dar = X[X['tower_region'] != 'Dar-es-salaam']
    # calc home tower as most frequently used (sender and rec) tower (for use on before moved df) outside the Dar region
    home_tower = not_Dar[['customer_id', 'tower_id']].groupby('customer_id')['tower_id'].apply(
        lambda x: x.mode()[0]).reset_index()
    home_tower.columns = ["ID","home_tower"]
    return home_tower



class CountCalls(object):
    def __init__(self, X, before_after='all'):
        self.X = X
        self.before_after_col = "Before_After"
        # self.X = add_before_after_col(self.X, new_col_name=self.before_after_col)
        self.IDs = pd.DataFrame(self.X['customer_id'].unique(), columns=['customer_id'])
        self.IDs["unique_id"] = np.nan
        self.before_after = before_after

        # calculate length of time period before moved
        first_date = self.X[['customer_id', 'timestamp']].groupby(by='customer_id', dropna=False).min().reset_index()
        first_date.columns = ['customer_id', 'first_date']
        self.X = self.X.merge(first_date, how='left', on="customer_id")
        self.X['days_before_dar'] = (self.X['move_date'] - self.X['first_date']).dt.days

        # calculate length of overall time period
        self.X['overall_days'] = self.X['days_before_dar'] + self.X['period_days']

        # get time period col to norm by
        if self.before_after == 'all':
            self.normed_col = 'overall_days'
        elif self.before_after == 'before':
            self.normed_col = 'days_before_dar'
        elif self.before_after == 'after':
            self.normed_col = 'period_days'
        else:
            print("for param before_after use only one of 'all', 'before', or 'after'")

    def count_calls(self, col_name, sent_or_rec='both', normed=True):
        """Counts number of calls either sent/rec or both.
        If Normed=True counts calls per day"""

        # sent and rec:
        if sent_or_rec == 'both':
            counts = self.X[['customer_id', self.normed_col,'unique_id']].groupby(by=['customer_id', self.normed_col]).count().reset_index()
        # sent:
        elif sent_or_rec == 'sent':
            counts = self.X[self.X['sender_receiver'] == "Sender"]
            counts = counts[['customer_id', self.normed_col,'unique_id']].groupby(by=['customer_id', self.normed_col]).count().reset_index()
        # rec:
        elif sent_or_rec == 'received':
            counts = self.X[self.X['sender_receiver'] == "Receiver"]
            counts = counts[['customer_id', self.normed_col,'unique_id']].groupby(by=['customer_id', self.normed_col]).count().reset_index()
        else:
            print("Use one of 'both','sent' or 'received'")

        if normed==True:
            counts['unique_id'] = round(counts['unique_id'] / counts[self.normed_col], 2)

        counts.drop(self.normed_col, inplace=True, axis=1)
        counts = counts.merge(self.IDs, how='right', on="customer_id")
        counts.drop('unique_id_y', axis=1, inplace=True)
        counts.columns = ["ID", col_name]
        counts[col_name].fillna(0, inplace=True)
        return counts


    def visited_dar_before_moved(self, colname):
        dict = {}
        for id in self.IDs['customer_id']:
            df_id = self.X[self.X['customer_id']==id]
            if 'Dar-es-salaam' in df_id['tower_region'].values:
                dict[id]=1
            else:
                dict[id]=0
        df = pd.DataFrame.from_dict(dict, orient='index').reset_index()
        df.columns = ['ID', colname]
        return df


    def where_from(self, colname, level):
        """finds most common 'ward' or 'region' that isn't Dar before moved"""
        level_col = 'tower_{}'.format(level)
        df = self.X[self.X['tower_region']!='Dar-es-salaam']
        df = df[['customer_id', level_col]].groupby('customer_id')[level_col].apply(
            lambda x: x.mode()).reset_index()
        df = df[['customer_id',level_col]]
        df.drop_duplicates(subset='customer_id', keep='first', inplace=True)
        df.columns = ['ID', colname]
        return df




class EntropyCalls(object):
    def __init__(self, X):
        self.X = X

    def entropy_calls(self, colname):
        d = {}
        for id in self.X['customer_id'].unique():
            called_ids = np.array(self.X['receiver_id'][self.X['customer_id'] == id])
            # calculate entropy:
            entropy = self.calc_entropy(called_ids)
            d[id] = entropy
        entropy_df = pd.DataFrame.from_dict(d, orient='index').reset_index()
        entropy_df.columns = ["ID", colname]
        return entropy_df


    def calc_entropy(self, X):
        value, counts = np.unique(X, return_counts=True)
        return round(entropy(counts), 4)


class CallingDistance(HDistance):
    def __init__(self, X):
        self.X = X
        self.IDs = self.X['customer_id'].unique()

    def calc_mean_dist(self, colname):
        df = self.X[['customer_id','calling_dist_km']].groupby('customer_id').mean().reset_index()
        df.columns = ['ID',colname]
        return df

    def calc_mode_dist(self, colname, x=20):
        # to nearest x km:
        self.X['round_calling_dist'] = round(self.X['calling_dist_km']/x)*x
        df = self.X[['customer_id','round_calling_dist']].groupby('customer_id')['round_calling_dist'].apply(
            lambda x: x.mode()[0]).reset_index()
        df.columns = ['ID',colname]
        return df

    def calc_dist_normed(self, colname, migration_distance, migrat_dist_col='H_dist_Mode'):
        """Calculates the mean calling distance divided by the distance migrated (calc using mode)"""
        df = self.X[['customer_id','calling_dist_km']].groupby('customer_id').mean().reset_index()
        df.columns = ['ID','Mean_Calling_Dist_After_Moving']
        df = df.merge(migration_distance, on='ID')
        df[colname] = round(abs(df[migrat_dist_col]-df['Mean_Calling_Dist_After_Moving']),4)
        df.drop(['Mean_Calling_Dist_After_Moving',migrat_dist_col], axis=1, inplace=True)
        return df

    def calls_to_Dar(self, colname):
        """Calculates the number of calls to Dar as a percentage of total calls
        (for feature use on before dataframe)."""
        dict = {}
        for id in self.IDs:
            df_id = self.X[self.X['customer_id'] == id]
            counts = df_id['called_region'][df_id['called_region']=='Dar-es-salaam'].shape[0]
            total_calls = df_id.shape[0]
            calls_to_Dar_perc = round((counts / total_calls) *100,2)
            dict[id] = calls_to_Dar_perc
        df = pd.DataFrame.from_dict(dict, orient='index').reset_index()
        df.columns = ['ID', colname]
        return df

    def calls_to_Dar_ward(self, colname, subwards, rank=True):
        """Calculates where the calls to Dar were most commonly to (which ward).
        If rank=True then the affluence rank of the ward is returned instead of the name of the ward
        (for feature use on before dataframe)."""
        dict = {}
        for id in self.IDs:
            df_id = self.X[self.X['customer_id'] == id]
            df_id = df_id[df_id['called_region']=='Dar-es-salaam']
            if df_id.shape[0] > 0:
                most_called_ward = df_id[['customer_id','called_ward']].groupby('customer_id')['called_ward'].apply(
                                   lambda x: x.mode()[0]).reset_index()['called_ward'].iloc[0]
                if rank==True:
                    ward = subwards[subwards['ward.name'] == most_called_ward]
                    result = round(ward['rank'].mean(), 2)
                elif rank==False:
                    result = most_called_ward
            else:
                if rank == True:
                    result = np.nan
                if rank == False:
                    result = ""
            dict[id] = result
        df = pd.DataFrame.from_dict(dict, orient='index').reset_index()
        df.columns = ['ID', colname]
        return df


    def calls_to_Dar_subward(self, colname, subwards, towers, rank=True):
        """Calculates where the calls to Dar were most commonly to (which tower and subward).
        If rank=True then the affluence rank of the ward (mean of sub-wards)
        is returned instead of the name of the ward
        (for feature use on before dataframe)."""
        dict = {}
        for id in self.IDs:
            df_id = self.X[self.X['customer_id'] == id]
            df_id = df_id[df_id['called_region']=='Dar-es-salaam']
            if df_id.shape[0] > 0:
                most_called_tower = df_id[['customer_id','called_tower']].groupby('customer_id')['called_tower'].apply(
                                   lambda x: x.mode()[0]).reset_index()['called_tower'][0]
                most_called_subward = towers[towers['tower_id'] == most_called_tower]['subward_id'].mean()
                if rank==True:
                    result = round(subwards['rank'][subwards['subward.id']==most_called_subward].mean(), 2)
                elif rank==False:
                    result = subwards['subward.name'][subwards["subward.id"] == most_called_subward]
            else:
                if rank == True:
                    result = np.nan
                if rank == False:
                    result = ""
            dict[id] = result
        df = pd.DataFrame.from_dict(dict, orient='index').reset_index()
        df.columns = ['ID', colname]
        return df


    def calc_number_of_home_calls(self, colname, home_tower_df):
        """count the % of calls were to the home tower
        (use on after df (df_call_dist_after) for feature)"""
        dict = {}
        for id in self.IDs:
            df_id = self.X[self.X['customer_id']==id]
            home_tower = home_tower_df['home_tower'][home_tower_df['ID']==id].mean()
            home_tower_calls = df_id['called_tower'][df_id['called_tower'] == home_tower]
            total_calls = df_id.shape[0]
            if (home_tower_calls.shape[0] > 0) == True:
                count_home_tower = home_tower_calls.shape[0]
            else:
                count_home_tower = 0
            home_tower_perc = round((count_home_tower / total_calls) * 100, 2)
            dict[id] = home_tower_perc
        df = pd.DataFrame.from_dict(dict, orient='index').reset_index()
        df.columns = ['ID', colname]
        df[colname].fillna(0, inplace=True)
        return df


def calls_at_home(X, colname, home_tower_df):
    """count the number of times sent/rec a call at the home tower,
    note: not necessarily separate 'visits' more a proxy for time visiting home
    (use on df_calls_after for feature)"""
    dict = {}
    for id in X['customer_id'].unique():
        df_id = X[X['customer_id'] == id]
        home_tower = home_tower_df['home_tower'][home_tower_df['ID'] == id].mean()
        home_tower_calls = df_id['tower_id'][df_id['tower_id'] == home_tower]
        if (home_tower_calls.shape[0] > 0) == True:
            count_home_tower = home_tower_calls.shape[0]
        else:
            count_home_tower = 0
        dict[id] = count_home_tower
    df = pd.DataFrame.from_dict(dict, orient='index').reset_index()
    df.columns = ['ID', colname]
    return df


class MobileMoney(object):
    def __init__(self, X):
        self.X = X

    def mean_balance(self, colname):
        df = self.X[['customer_id', 'pre_balance']].groupby(by='customer_id', as_index=False).mean()
        df.reset_index()
        df.columns = ["ID", colname]
        return df

    def had_mfs_before(self, colname):
        df = self.X[['customer_id', 'pre_balance']].groupby(by='customer_id', as_index=False).mean()
        df.reset_index()
        df[colname] = 0
        df[colname][df["pre_balance"]>=1] = 1
        df.drop("pre_balance", axis=1, inplace=True)
        df.columns = ["ID", colname]
        return df

    def count_transactions(self, colname):
        df = self.X[['customer_id', 'transfer_id']].groupby(by='customer_id', as_index=False).count()
        df.reset_index()
        df.columns = ["ID", colname]
        return df

class SumMoney(MobileMoney):
    def __init__(self, X):
        self.X = X
        first_date = self.X[['customer_id', 'transfer_date']].groupby(by='customer_id',
                                                                      dropna=False).min().reset_index()
        first_date.columns = ['customer_id', 'first_date']
        # join to mfs_df
        self.X = self.X.merge(first_date, how='left', on="customer_id")
        self.X['days_before_dar'] = (self.X['move_date'] - self.X['first_date']).dt.days

    def overall(self, colname, sender_rec="both", service_type='all'):
        """Calculate sum of either "Sender"/"Receiver"/"both" overall transactions
        (before and after move date) for all or a particular type of service"""

        if service_type != "all":
            X = self.X[self.X['service_type'] == service_type]
        else:
            X = self.X

        sum_trans = X[['customer_id', 'days_before_dar', 'length_full_period_days', 'sender_receiver',
                        'transaction_amount']].groupby(
            by=['customer_id', 'days_before_dar', 'length_full_period_days', 'sender_receiver'],
            dropna=False).sum().reset_index()

        sum_trans['overall_days'] = np.nan
        sum_trans['overall_days'][sum_trans['days_before_dar'] > 0] = sum_trans['days_before_dar'] + sum_trans['length_full_period_days']
        sum_trans['overall_days'][sum_trans['days_before_dar'] <= 0] = sum_trans['length_full_period_days']

        # create sum_trans dv normalised by days:
        if sender_rec=="both":
            sum_trans.drop("sender_receiver", axis=1, inplace=True)
            sum_trans = sum_trans[['customer_id', 'overall_days','transaction_amount']].groupby(by=['customer_id', 'overall_days']).sum().reset_index()
            sum_trans['sum_trans_div_days'] = sum_trans['transaction_amount'] / sum_trans['overall_days']
            sum_trans = sum_trans[["customer_id", "sum_trans_div_days"]]

        else:
            # create data with empty groups:
            sum_trans['sum_trans_div_days'] = sum_trans['transaction_amount'] / sum_trans['overall_days']
            sum_trans = sum_trans[["customer_id", "sender_receiver", "sum_trans_div_days"]]
            sum_trans = fill_empty_groups_overall(sum_trans)

        if sender_rec=="Sender":
            sum_trans = sum_trans[sum_trans['sender_receiver'] == 'Sender']
        if sender_rec=="Receiver":
            sum_trans = sum_trans[sum_trans['sender_receiver'] == 'Receiver']

        sum_trans = sum_trans[["customer_id", "sum_trans_div_days"]]
        sum_trans.columns = ["ID", colname]
        return sum_trans


    def before_after(self, colname, before_after, sender_rec="both", service_type = 'all'):
        """Calculate sum of either "Sender"/"Receiver"/"both" overall (before or after move date)
        for 'all' or a particular type of service"""
        if service_type != "all":
            X = self.X[self.X['service_type'] == service_type]
        else:
            X = self.X

        if before_after == 'before':
            before_after_col = 'days_before_dar'
        elif before_after == 'after':
            before_after_col = 'length_full_period_days'
        else:
            print("please enter 'before or 'after' for param before_after")

        sum_trans = X[['customer_id', before_after_col, 'sender_receiver',
                        'transaction_amount']].groupby(
            by=['customer_id', before_after_col, 'sender_receiver'],
            dropna=False).sum().reset_index()
        sum_trans = sum_trans[sum_trans[before_after_col] > 0]

        # create sum_trans dv normalised by days:
        if sender_rec=="both":
            sum_trans.drop("sender_receiver", axis=1, inplace=True)
            sum_trans = sum_trans[['customer_id', before_after_col,'transaction_amount']].groupby(
                by=['customer_id', before_after_col]).sum().reset_index()
            sum_trans['sum_trans_div_days'] = sum_trans['transaction_amount'] / sum_trans[before_after_col]

        else:
            # create data with empty groups:
            sum_trans['sum_trans_div_days'] = sum_trans['transaction_amount'] / sum_trans[before_after_col]
            sum_trans = sum_trans[["customer_id", "sender_receiver", "sum_trans_div_days"]]
            sum_trans = fill_empty_groups_overall(sum_trans)

        if sender_rec=="Sender":
            sum_trans = sum_trans[sum_trans['sender_receiver'] == 'Sender']
        if sender_rec=="Receiver":
            sum_trans = sum_trans[sum_trans['sender_receiver'] == 'Receiver']

        sum_trans = sum_trans[["customer_id", "sum_trans_div_days"]]
        sum_trans.columns = ["ID", colname]
        return sum_trans


def before(X, time_col="timestamp", before_after_col="Before_After"):
    X = add_before_after_col(X, new_col_name=before_after_col, date_col=time_col)
    X = X[X[before_after_col] == 'Before']
    return X

def after(X, time_col="timestamp", before_after_col="Before_After"):
    X = add_before_after_col(X, new_col_name=before_after_col, date_col=time_col)
    X = X[X[before_after_col] == 'After']
    return X


def calc_night_tower(X, x_perc):
    """Identifies the tower the person uses most often in Dar between
    the hours of 8pm and 8am after their move date"""
    Dar = X[X['tower_region'] == 'Dar-es-salaam']
    Dar.index = Dar['timestamp']
    night_counts = Dar.between_time('20:00','08:00')[['customer_id','tower_id','unique_id']].groupby(['customer_id','tower_id']).count().reset_index()
    all_counts = Dar[['customer_id','tower_id','unique_id']].groupby(['customer_id','tower_id']).count().reset_index()
    mode_tower = Dar[['customer_id','tower_id']].groupby('customer_id')['tower_id'].apply(
        lambda x: x.mode()[0]).reset_index()
    mode_night_tower = Dar.between_time('20:00','08:00')[['customer_id', 'tower_id']].groupby('customer_id')['tower_id'].apply(
        lambda x: x.mode()[0]).reset_index()

    mode_tower_counts = mode_tower.merge(all_counts, on=['customer_id','tower_id'], how='left')
    mode_night_tower_counts = mode_night_tower.merge(night_counts, on=['customer_id', 'tower_id'], how='left')
    dict = {}
    for id in Dar['customer_id'].unique():
        # if no night tower, then home tower == mode tower:
        if mode_night_tower_counts['tower_id'][mode_night_tower_counts['customer_id']==id].shape[0] < 1:
            dict[id] = int(mode_tower_counts['tower_id'][mode_tower_counts['customer_id'] == id])
        # if mode tower and mode night tower are the same select mode night tower:
        elif int(mode_tower_counts['tower_id'][mode_tower_counts['customer_id']==id]) == int(mode_night_tower_counts['tower_id'][mode_night_tower_counts['customer_id']==id]):
            dict[id] = int(mode_night_tower_counts['tower_id'][mode_night_tower_counts['customer_id']==id])
        # if mode tower overall and night mode tower are different...
        else:
            # night mode tower counts has to be > x_perc of overall mode tower counts to become a 'home tower',
            # otherwise it is overall mode (prevents random night visits becoming home towers)
            perc = round((int(mode_night_tower_counts['unique_id'][mode_night_tower_counts['customer_id']==id]) /
                    int(mode_tower_counts['unique_id'][mode_tower_counts['customer_id']==id])) * 100, 2)
            if perc > x_perc:
                dict[id] = int(mode_night_tower_counts['tower_id'][mode_night_tower_counts['customer_id'] == id])
            else:
                dict[id] = int(mode_tower_counts['tower_id'][mode_tower_counts['customer_id'] == id])
    return dict

def visit_university(df, label_df, colname):
    ids = pd.DataFrame(df['customer_id'].unique(), columns=['customer_id'])
    df_uni = ids.merge(label_df[['customer_id','University']], on='customer_id', how='left')
    df_uni.fillna(0, inplace=True)
    df_uni.columns = ['ID', colname]
    return df_uni


def ward_urban_or_rural(colname, df_urban_rural, df_ward_from, rural=1, urban=0):
    df_ward_from.columns = ["ID","ward_code"]
    df = df_ward_from.merge(df_urban_rural[['ward_code','rural_or_urban_ward']], how='left', on = 'ward_code')
    df.drop('ward_code', inplace=True, axis=1)
    df['rural_or_urban_ward'][df['rural_or_urban_ward'] == 'Rural'] = rural
    df['rural_or_urban_ward'][df['rural_or_urban_ward'] == 'Urban'] = urban
    df.columns = ['ID',colname]
    return df




























