import pandas as pd
from scipy.spatial import distance_matrix

def Calc_Lag(df, groupby, time_order, datetime_colname, lag_colname):
    df.sort_values(groupby+time_order, inplace=True, axis=0)
    df[datetime_colname] = df.apply(lambda r:
                                    pd.datetime.combine(r[time_order[0]],
                                                        r[time_order[1]]),1)
    df[lag_colname] = df.groupby(groupby)[
        datetime_colname].diff().dt.days
    return df



def Similariy_Matrix(df, cols):
    df_euclid = pd.DataFrame(
        1 / (1 + distance_matrix(df[cols], df[cols])),
        columns=cols, index=cols
    )
    return df_euclid