import pandas as pd
import numpy as np

def fill_empty_groups(df):
    id_list = list(df['customer_id'].unique())
    ids = []
    groups = []
    sender_rec = []
    for id in id_list:
        ids.append(id)
        ids.append(id)
        ids.append(id)
        ids.append(id)
        groups.append('Before')
        groups.append('Before')
        groups.append('After')
        groups.append('After')
        sender_rec.append('Receiver')
        sender_rec.append('Sender')
        sender_rec.append('Receiver')
        sender_rec.append('Sender')
    full = pd.DataFrame(list(zip(ids, groups, sender_rec)), columns=['customer_id', 'Before_After', 'sender_receiver'])
    df_full = full.merge(df, how='outer', left_on=['customer_id', 'Before_After', 'sender_receiver'],
                         right_on=['customer_id', 'Group', 'sender_receiver'])
    df_full['transaction_amount'] = df_full['transaction_amount'].replace(np.nan, 0)
    df_full['sum_trans_div_days'] = df_full['sum_trans_div_days'].replace(np.nan, 0)
    return df_full


def fill_empty_groups_overall(df):
    """No before and after cols"""
    id_list = list(df['customer_id'].unique())
    ids = []
    groups = []
    sender_rec = []
    for id in id_list:
        ids.append(id)
        ids.append(id)
        sender_rec.append('Receiver')
        sender_rec.append('Sender')
    full = pd.DataFrame(list(zip(ids, sender_rec)), columns=['customer_id', 'sender_receiver'])
    df_full = full.merge(df, how='outer', on=['customer_id', 'sender_receiver'])
    df_full['sum_trans_div_days'] = df_full['sum_trans_div_days'].replace(np.nan, 0)
    return df_full


def wide_to_long(df, a, b, group_col):
    a_df = pd.DataFrame(df[a])
    a_df[group_col] = a
    b_df = pd.DataFrame(df[b])
    b_df[group_col] = b
    df = pd.concat([a_df, b_df], axis=0)
    return df


def add_before_after_col(df, date_col='timestamp', move_date_col='move_date', new_col_name="Before_After"):
    df[new_col_name] = 'na'
    df[new_col_name][df[date_col] < df[move_date_col]] = 'Before'
    df[new_col_name][df[date_col] >= df[move_date_col]] = 'After'
    return df


def drop_neg_labels(df, df_labels):
    df_pos = df.merge(df_labels[['customer_id', 'Moved']], on='customer_id')
    df_pos = df_pos[df_pos['Moved'] == 1]
    return df_pos