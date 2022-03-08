import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import base64
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as st
import numpy as np
import datetime as dt

from adjustText import adjust_text
from matplotlib.lines import Line2D
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve

from Functions.Cluster import Normalise
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.seasonal import STL


def Plot_3D(df, save_path, x, y, move_date_col, region_col,
            type='scatter',
            uni_strict=True,
            n_graphs=10,
            c_uni=None,
            time_col='timestamp',
            id_col='customer_id',
            c_dar='m',
            c_not='c',
            id_count_col=None,
            Dist_Dict=None,
            show=True
            ):

    df[time_col] = pd.to_datetime(df[time_col], format="%Y-%m-%d %H:%M:%S")
    df[move_date_col] = pd.to_datetime(df[move_date_col], format="%Y-%m-%d %H:%M:%S")

    for id in df[id_col].unique()[0:n_graphs]:
        df_temp = df[df[id_col] == id]

        df_temp['first_date'] = min(df_temp [time_col])
        df_temp['last_date'] = max(df_temp [time_col])

        df_temp['time_from_first'] = (df_temp[time_col] - df_temp['first_date']).dt.total_seconds()

        # colour based on Dar or not:
        df_temp["colour"] = c_not
        df_temp["colour"][df_temp[region_col]=='Dar-es-salaam'] = c_dar

        # add colour for university:
        if c_uni:
            if uni_strict==True:
                df_temp["colour"][(df_temp['tower_id'] == 1604) |
                                  (df_temp['tower_id'] == 1619) |
                                  (df_temp['tower_id'] == 1640)
                ] = c_uni
            if uni_strict==False:
                df_temp["colour"][(df_temp['tower_id'] == 1604) |
                                  (df_temp['tower_id'] == 1619) |
                                  (df_temp['tower_id'] == 1640) |
                                  (df_temp['tower_id'] == 1611) |
                                  (df_temp['tower_id'] == 1664)
                                  ] = c_uni

        c = np.array(df_temp["colour"])
        X = np.array(df_temp[x])
        Y = np.array(df_temp[y])
        Z = np.array(df_temp['time_from_first'])

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        if type == 'scatter':

            # plot and add custom legend:
            if c_uni:
                ax.scatter3D(X[c!=c_uni], Y[c!=c_uni], Z[c!=c_uni], c=c[c!=c_uni], alpha=0.1)
                ax.scatter3D(X[c==c_uni], Y[c==c_uni], Z[c==c_uni], c=c[c==c_uni], alpha=1, marker='v', s=100)

                legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c_dar, markersize=15),
                                   Line2D([0], [0], marker='o', color='w', markerfacecolor=c_not, markersize=15),
                                   Line2D([0], [0], marker='v', color='w', markerfacecolor=c_uni, markersize=15,
                                          linestyle='None')]
                ax.legend(legend_elements, ['Dar', 'Not Dar', 'University'])

            else:
                ax.scatter3D(X, Y, Z, c=c, alpha=0.2)

                legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c_dar, markersize=15),
                                   Line2D([0], [0], marker='o', color='w', markerfacecolor=c_not, markersize=15)]
                ax.legend(legend_elements, ['Dar', 'Not Dar'])

        if type == 'line':
            ax.plot(X, Y, Z, alpha=0.6)

        # plot plane for move date:
        move_date_value = pd.DataFrame((df_temp[move_date_col] - df_temp['first_date']).dt.total_seconds()).mean()[0]
        x_mesh = np.linspace(np.amin(X), np.amax(X), 100)
        y_mesh = np.linspace(np.amin(Y), np.amax(Y), 100)
        xx, yy = np.meshgrid(x_mesh, y_mesh)
        zz = move_date_value * np.ones(xx.shape)
        ax.plot_surface(xx, yy, zz, alpha=0.2, color='deeppink')

        # add axis lables:
        ax.set_xlabel('Lon')
        ax.set_ylabel('Lat')
        ax.set_zlabel('Time (sec)')
        plt.title(str(df_temp[id_count_col].iloc[0]) + ": " + id)

        # Plot distance markers for error analysis:
        if bool(Dist_Dict):
            cs=['r','b']
            for dict, c in zip(Dist_Dict, cs):
                # not dar
                d = Dist_Dict[dict]
                ax.plot(xs=np.ones(Z.shape[0]) * float(d.Lon_Not),
                        ys=np.ones(Z.shape[0]) * float(d.Lat_Not),
                        zs=np.linspace(np.amin(Z), np.amax(Z), Z.shape[0]), c=c)

                ax.plot(xs=np.ones(Z.shape[0]) * float(d.Lon_Dar),
                        ys=np.ones(Z.shape[0]) * float(d.Lat_Dar),
                        zs=np.linspace(np.amin(Z), np.amax(Z), Z.shape[0]), c=c)

        # save and close:
        plt.tight_layout()
        if id_count_col:
            id_num = df_temp[id_count_col].iloc[0]
            fig.savefig(str(save_path) + str(id_num) + '_' + id + '.png')
        else:
            fig.savefig(str(save_path) + id + '.png')
        if show==True:
            plt.show()
    return



def plot_hist(save_name, x, bins, save_path, title, xlim=None, ylim=None,
              fig_size=(20,20), xlab="", ylab='', fontsize=10):
    """Plots a histogram and saves to file. Also outputs a html image tag"""
    plt.figure(figsize=fig_size)
    if xlim != None:
        plt.xlim(xlim)
    if ylim != None:
        plt.ylim(ylim)
    plt.hist(x, bins = bins, color ="skyblue", alpha=0.5)
    plt.title(title, fontsize=fontsize)
    plt.xlabel(xlab, fontsize=fontsize)
    plt.ylabel(ylab, fontsize=fontsize)
    plt.axvline(x=np.mean(x) - np.std(x), ls="--", color='#2ca02c', alpha=0.7)
    plt.axvline(x=np.mean(x) + np.std(x), ls="--", color='#2ca02c', alpha=0.7)
    plt.axvline(x=np.mean(x), ls="-", color='red', alpha=0.7)
    plt.axvline(x=np.percentile(x, 5), ls="dotted", color='lightblue', alpha=0.7)
    plt.axvline(x=np.percentile(x, 95), ls="dotted", color='lightblue', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path+save_name+".png")
    data_uri = base64.b64encode(open(save_path+save_name+".png", 'rb').read()).decode('utf-8')
    image_tag = '<img src="data:image/png;base64,{0}">'.format(data_uri)
    return image_tag



def plot_hist_stacked(save_name, x1, x2, bins, save_path,
                      x1_lab, x2_lab, title, title_fontsize = 10,
                      xlab = "", ylab="",
                      xlim=None, fig_size=(20,20)):
    """Plots a histogram and saves to file."""
    plt.figure(figsize=fig_size)
    if xlim != None:
        plt.xlim(xlim)
    plt.hist(x1, bins=bins, stacked=True, density=False, color ="skyblue", label=x1_lab, alpha=0.5)
    plt.hist(x2, bins=bins, stacked=True, density=False, color ="blue", label=x2_lab, alpha=0.3)
    plt.title(title,fontsize=title_fontsize)
    plt.legend()
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.tight_layout()
    plt.savefig(save_path+save_name+".png")
    return



def plot_reg(x, y, df, save_path, lab_name, point_lab_size=10,
             adjust=False, labels=False, lab_perc=None, tick_lab_size=11,
             xjit=None, yjit=None, ci=95, fig_size = (20,20)):
    """Plots a regression plot with results printed beneath and distributions at the side.
    Options to add labels (type) and whether to just include outliers or not."""

    if isinstance(y, str):
        y = [y]

    if (adjust == True) and (labels == True):
        raise Exception("Both 'adjust' and 'labels' can not be True. "
                        "Choose either normal labels with labels=True "
                        "or adjusted labels with adjust=True.")
    for var in y:
        # only label the outliers
        if lab_perc:
            p = np.percentile(df[var], lab_perc)
            label = []
            for i in df[var]:
                if i >= p:
                    label.append('True')
                else:
                    label.append('False')
            df['Label'] = label
            labs = list(df[lab_name][df['Label'] == 'True'])
            xp = pd.DataFrame(df[x][df['Label'] == 'True']).reset_index(drop=True)
            yp = pd.DataFrame(df[var][df['Label'] == 'True']).reset_index(drop=True)
        else:
            labs = list(df[lab_name])
            xp = pd.DataFrame(df[x]).reset_index(drop=True)
            yp = pd.DataFrame(df[var]).reset_index(drop=True)

        # perform regression/corr
        slope, intercept, r_value, p_value, std_err = st.linregress(x=df[x], y=df[var])
        r_squared = r_value ** 2

        with pd.option_context('display.float_format', '{:0.2f}'.format):
            print("sum of transfers \n \n"
                  "R^2={:.2f}, r={:.2f}, p-value={:.4f}, Std.Err={:.2f}, \n"
                  "Slope={:.2f}, Intercept={:.2f}, CI's = {}%".format(
                    r_squared, r_value, p_value, std_err, slope, intercept, ci),
                    file=open(str(save_path) + var + '_Results.txt', "w"))

            if yjit and yjit > 0:
                      print(" \n \n*note jitter has been added to plot to aid visualisation \n"
                            "(xjit={}, yjit={})".format(xjit, yjit),
                    file=open(str(save_path) + var + '_Results.txt', "a"))


            plt.figure(figsize=fig_size)
            g = sns.jointplot(x=x, y=var, data=df, kind='reg',
                              joint_kws={'line_kws': {'color': 'red', "ls": ':'}})

            f = open(str(save_path) + var + '_Results.txt')
            text = f.read()
            ylab = var.replace("_", " ")
            g.set_axis_labels(text, ylab, fontsize=tick_lab_size)
            plt.xticks(fontsize=tick_lab_size)
            plt.subplots_adjust(left=0.2, bottom=0.3)

            if labels == True:
                label_point(xp, yp, pd.DataFrame(labs)[0], plt.gca(), point_lab_size)
                plt.savefig(str(save_path) + var + "_Plot_Labels.png")

            if adjust == True:
                xp=np.array(xp.iloc[:,0])
                yp=np.array(yp.iloc[:,0])
                labs=np.array(labs)

                texts = [plt.text(xp[index], yp[index], labs[index]) for index in range(0, len(xp))]
                adjust_text(texts, arrowprops=dict(arrowstyle='-', color='black'))
                plt.savefig(str(save_path) + var + "_Plot_Adjust.png")
            else:
                plt.savefig(str(save_path) + var + "_Plot.png")

            plt.clf()
            plt.cla()
            plt.close()
    return


def label_point(x, y, val, ax, lab_size):
    a = pd.concat([x,y,val], axis=1)
    a.columns = ['x','y','val']
    for i, point in a.iterrows():
        ax.text(point['x']+.02, point['y'], str(point['val']), size=lab_size)



def PlotOverallGroup(df, x, y, hue, save_path, save_name,
                     figsize, kind='bar', ci=None,
                     title=" ", y_min=None,
                     yticks=None, ylabs=None, legend=False,
                     legend_loc='upper right',
                     title_fontsize=10, x_tick_fontsize=10,
                     xlab= " ", ylab= " ", bw=.1,
                     estimator=np.mean):
    if estimator == np.mean:
        line = df[y].mean()
        linestyle = '--'
    if estimator == np.median:
        line = df[y].median()
        linestyle = '--'
    else:
        line = 0
        linestyle = ''
    df.sort_values(by=hue, axis=0, inplace=True, ascending=False)
    sns.set_palette("Set2")
    plt.figure(figsize=figsize)
    if kind == 'violin':
        chart = sns.catplot(x=x, y=y, hue=hue, data=df, kind=kind,
                            legend=False, ci=ci, bw=bw, cut=y_min)
    else:
        chart = sns.catplot(x=x, y=y, hue=hue, data=df, kind=kind,
                            legend=False, ci=ci, estimator=estimator)

    chart.set_xticklabels(rotation=45, horizontalalignment='center')
    plt.xticks(fontsize=x_tick_fontsize)
    chart.set(xlabel = xlab, ylabel= ylab)
    plt.axhline(y=line, color='grey', linestyle=linestyle)
    if y_min:
        plt.ylim(y_min)
    if yticks and ylabs:
        plt.yticks(ticks=yticks, labels=ylabs)
    if legend == True:
        chart.add_legend(loc=legend_loc)
    plt.subplots_adjust(left=0.3)
    plt.title(title, y=1, fontsize = title_fontsize)
    plt.tight_layout()
    plt.savefig(save_path+save_name+".png")
    plt.clf()
    plt.cla()
    plt.close()
    return


def Plot_Time_Series(df, date_col, count_col, grouper,
                     save_path, save_name,
                     ylab,
                     scale=False,
                     time_aggregate='W-MON',
                     fig_size=(20,20)):
    df = Aggregate_Time_Count(count_col, date_col, df, grouper, scale, time_aggregate)
    plt.figure(figsize=fig_size)
    sns.lineplot(data=df, x=date_col, y=count_col, hue=grouper)
    plt.ylabel(ylab)
    plt.xlabel(date_col.replace("_", ' '))
    plt.tight_layout()
    plt.savefig(save_path+save_name+".png")
    plt.clf()
    plt.cla()
    plt.close()
    return df


def Aggregate_Time_Count(count_col, date_col, df, grouper, scale, time_aggregate):
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.groupby([grouper, pd.Grouper(key=date_col, freq=time_aggregate)])[count_col].size(). \
        reset_index(). \
        sort_values([grouper, date_col])
    if scale == True:
        for case in list(df[grouper].unique()):
            df[df[grouper] == case] = Normalise(df[df[grouper] == case])
    return df


def Aggregate_Time_Sum(sum_col, date_col, df, grouper, scale, time_aggregate):
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.groupby([grouper, pd.Grouper(key=date_col, freq=time_aggregate)])[sum_col].sum(). \
        reset_index(). \
        sort_values([grouper, date_col])
    if scale == True:
        for case in list(df[grouper].unique()):
            df[df[grouper] == case] = Normalise(df[df[grouper] == case])
    return df

def Aggregate_Time_Mean(sum_col, date_col, df, grouper, scale, time_aggregate):
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.groupby([grouper, pd.Grouper(key=date_col, freq=time_aggregate)])[sum_col].mean(). \
        reset_index(). \
        sort_values([grouper, date_col])
    if scale == True:
        for case in list(df[grouper].unique()):
            df[df[grouper] == case] = Normalise(df[df[grouper] == case])
    return df


def Time_Series_Decomposition(y, model_type, save_path,
                              save_name, title, period=None):
    """Plots decomposition of time series data.
    If period is not 1, then y must be a pandas object with a timeseries index.
    Model type options are strings ['additive','multiplicative','STL']"""
    if model_type == 'STL':
        result = STL(y, period=period).fit()
    else:
        result = seasonal_decompose(x=y, model=model_type, period=period)

    fig = result.plot()
    plt.tight_layout()
    axes = fig.axes

    for i, ax in enumerate(axes):
        if i == 3:
            labels = ax.get_xticklabels()
            ax.set_xticklabels(labels, rotation=90, fontsize=7)
        else:
            ax.set_xticklabels([])
    axes[0].set_title(" ")
    fig.suptitle(title)
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(save_path + save_name + ".png")
    plt.clf()
    plt.cla()
    plt.close()
    return


def plot_scatter(x, y, df, save_path, lab_name, point_lab_size=10,
             adjust=False, labels=False, lab_perc=None, tick_lab_size=16,
             xjit=None, yjit=None, ci=95, fig_size = (20,20), legend=True):
    """Plots a scatter plot with results printed beneath and distributions at the side.
    Options to add labels (type) and whether to just include outliers or not."""

    if isinstance(y, str):
        y = [y]

    if (adjust == True) and (labels == True):
        raise Exception("Both 'adjust' and 'labels' can not be True. "
                        "Choose either normal labels with labels=True "
                        "or adjusted labels with adjust=True.")
    for var in y:
        df.dropna(subset= [var], inplace=True, axis=0)
        # only label the outliers
        if lab_perc:
            p = np.percentile(df[var], lab_perc)
            label = []
            for i in df[var]:
                if i >= p:
                    label.append('True')
                else:
                    label.append('False')
            df['Label'] = label
            labs = list(df[lab_name][df['Label'] == 'True'])
            xp = pd.DataFrame(df[x][df['Label'] == 'True']).reset_index(drop=True)
            yp = pd.DataFrame(df[var][df['Label'] == 'True']).reset_index(drop=True)
        else:
            labs = list(df[lab_name])
            xp = pd.DataFrame(df[x]).reset_index(drop=True)
            yp = pd.DataFrame(df[var]).reset_index(drop=True)

        plt.figure(figsize=fig_size)
        palette="coolwarm"
        g = sns.scatterplot(x=x, y=var, data=df, hue=var, size=var, legend=legend, palette=palette)

        ylab = var.replace("_", " ")
        xlab = x.replace("_", " ")
        g.set_xlabel(xlab, fontsize=tick_lab_size)
        g.set_ylabel(ylab, fontsize=tick_lab_size)

        # plt.xticks(fontsize=tick_lab_size)
        plt.subplots_adjust(left=0.2, bottom=0.3)

        if labels == True:
            label_point(xp, yp, pd.DataFrame(labs)[0], plt.gca(), point_lab_size)
            plt.tight_layout()
            plt.savefig(str(save_path) + var + "_Plot_Labels.png")

        if adjust == True:
            xp=np.array(xp.iloc[:,0])
            yp=np.array(yp.iloc[:,0])
            labs=np.array(labs)

            texts = [plt.text(xp[index], yp[index], labs[index]) for index in range(0, len(xp))]
            adjust_text(texts, arrowprops=dict(arrowstyle='-', color='black'))
            plt.tight_layout()
            plt.savefig(str(save_path) + var + "_Plot_Adjust.png")
        else:
            plt.tight_layout()
            plt.savefig(str(save_path) + var + "_Plot.png")

        plt.clf()
        plt.cla()
        plt.close()
    return


def Plot_ANOVA(df, x, y, save_path, group, labs, save_name ="Count_positive_all_questions",
               fontsize=12, legendloc="upper left",
               xorder_asc=False, fig_size=(8,8),
               ylab=" "):
    df.sort_values(by=[x, group], ascending=xorder_asc, inplace=True, axis=0)

    plt.figure(figsize=(25, 15))
    rc = {'axes.labelsize': fontsize, 'font.size': fontsize, 'legend.fontsize': fontsize}
    sns.set(rc=rc)
    plt.figure(figsize=fig_size)
    cmap = plt.get_cmap('RdYlGn').reversed()
    matplotlib.cm.register_cmap("mycolormap", cmap)
    pallette = sns.color_palette("mycolormap", 2)

    chart = sns.catplot(x=x, y=y, hue=group, data=df, kind='bar',
                        legend=False, ci=95, palette=pallette)

    chart.set_xticklabels(rotation=45, horizontalalignment='right')
    plt.title("")
    plt.xticks(fontsize=fontsize)
    chart.set(xlabel = "", ylabel=ylab)
    # plt.legend(title='', loc=legendloc, labels=labs)
    plt.tight_layout()
    plt.savefig(save_path+save_name+".png")
    plt.clf()
    plt.cla()
    plt.close()
    return


def plot_precis_recall_curve(yhat, y_test, save_path, save_name, y, model_lab):
    pos_probs = yhat[:, 1]
    no_skill = len(y[y == 1]) / len(y)
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    precision, recall, _ = precision_recall_curve(y_test, pos_probs)
    plt.plot(recall, precision, marker='.', label=model_lab)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.savefig(save_path + save_name + ".png")


def plot_precis_recall_curve_compare(yhat1, yhat2, y_test, save_path, save_name, y, model_lab1, model_lab2):
    pos_probs1 = yhat1[:, 1]
    pos_probs2 = yhat2[:, 1]
    no_skill = len(y[y == 1]) / len(y)
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    precision, recall, _ = precision_recall_curve(y_test, pos_probs1)
    plt.plot(recall, precision, marker='.', label=model_lab1)
    precision, recall, _ = precision_recall_curve(y_test, pos_probs2)
    plt.plot(recall, precision, marker='.', label=model_lab2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.savefig(save_path + save_name + ".png")
    plt.clf()
    plt.cla()
    plt.close()


def plot_roc(probs, y_test, save_path, save_name, model_name):
    # keep probabilities for the positive outcome only
    lr_probs = probs[:, 1]
    ns_probs = [0 for _ in range(len(y_test))]
    # calculate scores
    ns_auc = roc_auc_score(y_test, ns_probs)
    lr_auc = roc_auc_score(y_test, lr_probs)
    # summarize scores
    print('No Skill: ROC AUC=%.3f' % (ns_auc))
    print('{}: ROC AUC=%.3f'.format(model_name) % (lr_auc))
    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
    # plot the roc curve for the model
    plt.figure(figsize=(6, 6))
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    plt.plot(lr_fpr, lr_tpr, marker='.', label=model_name)
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()
    # show the plot
    plt.savefig(save_path + save_name + ".png")
    plt.clf()
    plt.cla()
    plt.close()

def plot_roc_compare(probs1, probs2, y_test, save_path, save_name,
                     model1_name, model2_name, no_skill=False):
    # keep probabilities for the positive outcome only
    probs1 = probs1[:, 1]
    probs2 = probs2[:, 1]
    ns_probs = [0 for _ in range(len(y_test))]
    # calculate scores
    ns_auc = roc_auc_score(y_test, ns_probs)
    auc1 = roc_auc_score(y_test, probs1)
    auc2 = roc_auc_score(y_test, probs2)
    # summarize scores
    print('No Skill: ROC AUC=%.3f' % (ns_auc))
    print('{}: ROC AUC=%.3f'.format(model1_name) % (auc1))
    print('{}: ROC AUC=%.3f'.format(model2_name) % (auc2))
    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
    fpr1, tpr1, _ = roc_curve(y_test, probs1)
    fpr2, tpr2, _ = roc_curve(y_test, probs2)
    # plot the roc curve for the model
    plt.figure(figsize=(6, 6))
    if no_skill==True:
        plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    plt.plot(fpr1, tpr1, marker='.', label=model1_name)
    plt.plot(fpr2, tpr2, marker='.', label=model2_name)
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()
    # show the plot
    plt.savefig(save_path + save_name + ".png")
    plt.clf()
    plt.cla()
    plt.close()