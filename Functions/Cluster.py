import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm
from sklearn.decomposition import NMF, PCA, KernelPCA


def PCA_plot2D(X, cols, save_path, save_name, hue, jitter=None):
    X.sort_values(by=hue,inplace=True,ascending=False)
    # PCA
    hue = [hue]
    pca = PCA(n_components=2)
    X2D = pca.fit_transform(X[cols])
    plt.figure(figsize=(7, 5))
    if jitter:
        x=jitter_funct(X2D[:, 0], jitter)
        y=jitter_funct(X2D[:, 1], jitter)
    else:
        x=X2D[:, 0]
        y=X2D[:, 1]
    sns.scatterplot(x=x,
                    y=y,
                    hue=np.array(X[hue])[:, 0],
                    s=20,
                    alpha=0.5)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.savefig(str(save_path) + save_name + ".png")
    plt.clf()
    plt.cla()
    plt.close()
    return


def PCA_plot2D_group(X, cols, save_path, save_name, hue, id,
                     size_dict, marker_dict, size_order_list):
    # PCA
    X.reset_index(inplace=True, drop=True)
    pca = PCA(n_components=2)
    X2D = pca.fit_transform(X[cols])

    x=pd.DataFrame(X2D[:, 0],columns=['x'])
    y=pd.DataFrame(X2D[:, 1], columns=['y'])

    df = X[[id, hue]].join(x, how='left')
    df = df.join(y, how='left')

    df.sort_values(by=hue, inplace=True, ascending=False)

    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=df,
                    x='x',
                    y='y',
                    hue=hue,
                    style=hue,
                    size=hue,
                    sizes=size_dict,
                    size_order=size_order_list,
                    alpha=1,
                    markers=marker_dict)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.savefig(str(save_path) + save_name + ".png")
    plt.clf()
    plt.cla()
    plt.close()
    return


def Normalise(df):
    df_num = df.select_dtypes(include=[np.number])
    for feature_name in df_num.columns:
        max_value = df_num[feature_name].max()
        min_value = df_num[feature_name].min()
        df_num[feature_name] = (df_num[feature_name] - min_value) / (max_value - min_value)
    df[df.select_dtypes(include=[np.number]).columns] = df_num
    return df


def jitter_funct(values,jitter):
    return values + np.random.normal(jitter,0.1,values.shape)


def PCA_fit(df, cols, save_path, save_name):
    """Performs PCA and saves the fit graph for number of components"""
    pca = PCA().fit(df[cols])
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.subplots_adjust(left=0.1)
    plt.savefig(str(save_path) + save_name + ".png")
    plt.clf()
    plt.cla()
    plt.close()
    return



def Fill_nas(cols, df, id):
    # fill NAs with mean of group (id)
    df = df[[id] + cols]
    na_cols = df.columns[df.isna().any()].tolist()
    for col in na_cols:
        df[col] = df.groupby(id)[col].transform(lambda x: x.fillna(x.mean()))
    # remove where only one transfer (na)
    df.dropna(inplace=True)
    return df



def Remove_Outliers(df, cols, lower=.00, upper=.99):
    for col in cols:
        removed_outliers = df[col].between(df[col].quantile(lower),
                                           df[col].quantile(upper))
        print(col + ' : ' +str(df[col][removed_outliers].size) + "/" + str(df.shape[0]) + " data points remain.")
        drops = df[col][~removed_outliers].index
        df[col].drop(drops, inplace=True)
    return df



def Interpret_PCs(X, cols, id, n_comp, save_path, seed=9393):
    pca_comp = PCA(n_components=n_comp, random_state=seed, svd_solver='full', tol=0.05)
    pca_comp.fit(X[cols])
    color = cm.inferno_r(np.linspace(.2, .8, 5))
    d = {}
    for i in range(0, pca_comp.n_components_):
        d[i] = (pca_comp.components_[i])
        d[i] = pd.Series(data=d[i], index=X[cols].columns)
        d[i] = pd.DataFrame(d[i])
        d[i]['abs'] = abs(d[i].iloc[:, 0])
        num = i + 1
        d[i].rename(columns={0: 'PC' + str(num)}, inplace=True)
        d[i].sort_values(by='abs', ascending=True, inplace=True)
        d[i]['Var'] = d[i].index
        d[i] = d[i].iloc[len(d[i]) - 10:len(d[i]), :]
        plt.barh(d[i]['Var'].iloc[0:10], d[i].iloc[0:10, 0], color=color)
        plt.xticks(rotation=90)
        plt.suptitle('PC' + str(num))
        plt.subplots_adjust(left=0.4)
        plt.savefig(save_path + 'PC' + str(num) + ".png")
        plt.clf()
        plt.cla()
        plt.close()
    return


