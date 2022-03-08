import scipy.stats as st
import datetime
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
import statistics as s
import statsmodels.formula.api as smf
import matplotlib.font_manager
import math
from statsmodels.formula.api import ols
from datetime import datetime
from scipy.stats import chi2_contingency
from scipy.stats import t

def bonferroni_thresh(alpha_fwe, n):
    return alpha_fwe / n


def t_test(df, a, b, save_path, group_col, dv, fig_size,
           test_type, save_name, title, data_shape,
           box=False, tick_size=10, subset=None, aspect=1,
           bon_corr_pthresh=None, ylab= " ", xlab= " ", title_size=18,
           xticklabs = 'get', fontsize=14):
    """ Performs a t-test and outputs plot. Options for type of test
    and data shape. Na's treated accordingly.
    The results of the t-test is printed below the graph on the x axis."""
    if subset== None:
        subset=dv
    if data_shape == 'long':
        if test_type == 'paired':
            # drop nas and then drop if only one customer_id:
            df.dropna(inplace=True, axis=0, subset=[subset])
            df = df[df.duplicated('customer_id', keep=False)]
            # perform t-test:
            t, p = st.ttest_rel(a=df[dv][df[group_col] == a], b=df[dv][df[group_col] == b], axis=0,
                                nan_policy='raise')

        if test_type == 'independent':
            t, p = st.ttest_ind(a=df[dv][df[group_col] == a], b=df[dv][df[group_col] == b], axis=0, equal_var=True,
                                nan_policy='omit')

    if data_shape == 'wide':
        if test_type == 'paired':
            df.dropna(inplace=True, axis=0, subset=[subset])
            t, p = st.ttest_ind(a=df[a], b=df[b], axis=0, equal_var=True,
                                nan_policy='raise')

        if test_type == 'independent':
            t, p = st.ttest_ind(a=df[a], b=df[b], axis=0, equal_var=True,
                                nan_policy='omit')

        # convert data to long form for plotting:
        a_df = pd.DataFrame(df[a])
        a_df[group_col] = a
        b_df = pd.DataFrame(df[b])
        b_df[group_col] = b
        df = pd.concat([a_df, b_df], axis=0)

    if (p.dtype == 'float') and (t.dtype == 'float'):
        t_string, p_string = f'{t:.4f}', f'{p:.4f}'

        if p < 0.001:
            print("t= " + t_string + ", p= " + p_string + " *** ",
                  file=open(str(save_path) + str(save_name) + '.txt', "w"))
        elif p < 0.01:
            print("t= " + t_string + ", p= " + p_string + " ** ",
                  file=open(str(save_path) + str(save_name) + '.txt', "w"))
        elif p < 0.05:
            print("t= " + t_string + ", p= " + p_string + " * ",
                  file=open(str(save_path) + str(save_name) + '.txt', "w"))
        else:
            print("t= " + t_string + ", p= " + p_string,
                  file=open(str(save_path) + str(save_name) + '.txt', "w"))

    # Read test results
    f = open(str(save_path) + str(save_name) + ".txt")
    text = f.read()

    # plot:
    rc = {'axes.labelsize': fontsize, 'font.size': fontsize, 'legend.fontsize': fontsize, 'axes.titlesize': title_size}
    sns.set(rc=rc)
    plt.figure(figsize=fig_size)
    cmap = plt.get_cmap('RdYlGn').reversed()
    matplotlib.cm.register_cmap("mycolormap", cmap)
    palette = sns.color_palette("mycolormap", 2)
    ax = sns.catplot(x=group_col, y=dv, data=df, palette=palette, order=[a,b], ci=95, aspect=aspect, kind='bar')
    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)
    ax.set(ylabel=ylab)
    if xticklabs == 'get':
        ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right')
    else:
        ax.set_xticklabels(xticklabs, rotation=40, ha='right')
    if bon_corr_pthresh != None and p <= bon_corr_pthresh:
        ax.set(xlabel= xlab + "\n \n" +
                      text +
                      "\n (sig. at the Bonferroni corrected p threshold (p<{}))".format(bon_corr_pthresh))
    if bon_corr_pthresh != None and p > bon_corr_pthresh:
        ax.set(xlabel= xlab + "\n \n" +
                      text +
                      "\n (not sig. at the Bonferroni corrected p threshold (p<{}))".format(bon_corr_pthresh))
    else:
        ax.set(xlabel=xlab)
    plt.title(title, fontsize=title_size)
    plt.subplots_adjust(bottom=0.5, left=0.1)
    plt.tight_layout()
    plt.savefig(str(save_path) + str(save_name))

    if box==True:
        plt.figure(figsize=fig_size)
        cmap = plt.get_cmap('RdYlGn').reversed()
        matplotlib.cm.register_cmap("mycolormap", cmap)
        palette = sns.color_palette("mycolormap", 2)
        ax = sns.boxplot(x=group_col, y=dv, data=df, palette=palette, order=[a, b])
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        ax.set(ylabel=ylab)
        if xticklabs == 'get':
            ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right')
        else:
            ax.set_xticklabels(xticklabs, rotation=40, ha='right')
        if bon_corr_pthresh != None and p <= bon_corr_pthresh:
            ax.set(xlabel= xlab + "\n \n" +
                          text +
                          "\n (sig. at the Bonferroni corrected p threshold (p<{}))".format(bon_corr_pthresh))
        if bon_corr_pthresh != None and p > bon_corr_pthresh:
            ax.set(xlabel= xlab + "\n \n" +
                          text +
                          "\n (not sig. at the Bonferroni corrected p threshold (p<{}))".format(bon_corr_pthresh))
        plt.title(title, fontsize=title_size)
        plt.subplots_adjust(bottom=0.5, left=0.1)
        plt.tight_layout()
        plt.savefig(str(save_path) + str(save_name) + "_Boxplot.png")
    return



def PlotChiSquare(df, x, count_col,
                  hue, save_path, save_name,
                  after_only,
                  plot_counts_greater_than=None,
                  before_after_col= 'Group',
                  figsize= (8,8),
                  kind='bar', ci=None,
                  title=" ", y_min=None,
                  yticks=None, ylabs=None, legend=False,
                  legend_loc='upper right',
                  title_fontsize=10, x_tick_fontsize=10,
                  xlab= " ", ylab= " ", bw=.1,
                  estimator=np.mean,
                  in_out_breakdown=None,
                  aspect=1,
                  bb=(1.01, 1),
                  standardise=None,
                  id_col='customer_id',
                  pallette="Set2"):

    if after_only==True:
        if in_out_breakdown:
            df = df[[x] + [before_after_col] + [in_out_breakdown] + [hue] + [count_col]].groupby(
                [x] + [before_after_col] + [in_out_breakdown] + [hue]).count()
            df.reset_index(inplace=True)
            df = df[df[before_after_col] == 'After']
            hue = [hue] + [in_out_breakdown]

        else:
            df = df[[x]+[before_after_col]+[hue]+[count_col]].groupby(
                [x]+[before_after_col]+[hue]).count()
            df.reset_index(inplace=True)
            df = df[df[before_after_col] == 'After']

    if after_only==False:
        if in_out_breakdown:

            if standardise:
                df = Standardise(count_col, df, hue, id_col, standardise, x,
                                 in_out_breakdown=in_out_breakdown)

            else:
                df = df[[x] + [in_out_breakdown] + [hue] + [count_col]].groupby(
                    [x] + [in_out_breakdown] + [hue]).count()
                df.reset_index(inplace=True)

            hue = [hue] + [in_out_breakdown]

        else:
            if standardise:
                df = Standardise(count_col, df, hue, id_col, standardise, x)

            else:
                df = df[[x]+[hue]+[count_col]].groupby(
                    [x]+[hue]).count()
                df.reset_index(inplace=True)

    if plot_counts_greater_than:
        df_plot = df[df[count_col]>plot_counts_greater_than]
    else:
        df_plot = df.copy()

    ChiSquareTest(count_col, df, hue, save_name, save_path, x)

    if estimator == np.mean:
        line = df[count_col].mean()
        linestyle = '--'
    if estimator == np.median:
        line = df[count_col].median()
        linestyle = '--'
    else:
        line = 0
        linestyle = ''

    if isinstance(hue, list):
        order = hue
        order.append(x)
    else:
        order = [hue]
        order.append(x)
    df_plot.sort_values(by=order, axis=0, inplace=True, ascending=False)

    if in_out_breakdown:
        hue = hue[0]
        df_plot[hue] = df_plot[hue].astype(str) + ' - ' + df_plot[in_out_breakdown].astype(str)

    PlotChi(aspect, bb, bw, ci, count_col, df_plot, estimator, figsize, hue, kind, legend, legend_loc, line, linestyle,
            save_name, save_path, title, title_fontsize, x, x_tick_fontsize, xlab, y_min, ylab, ylabs, yticks, pallette)
    return


def ChiSquareTest(count_col, df, hue, save_name, save_path, x):
    df_wide = df.pivot_table(index=x, columns=hue, values=count_col)
    df_wide.to_csv(save_path + save_name + "_Raw_counts.csv", index=True)
    df_wide.dropna(axis=1, inplace=True)
    res = pd.DataFrame(chi2_contingency(df_wide))
    res = res.transpose()
    res.columns = ["Pearson's Chi-sq indep (X^2)", "p-value", "(df)", "Expected values"]
    res.drop("Expected values", axis=1, inplace=True)
    res["p-value"] = res["p-value"].astype(float).round(4)
    res["(df)"] = "(" + str(res["(df)"][0]) + ")"
    with pd.option_context('display.float_format', '{:0.4f}'.format):
        print(res.to_string(index=False),
              file=open(str(save_path) + str(save_name) + '.txt', "w"))


def Standardise(count_col, df, hue, id_col, standardise, x, in_out_breakdown=None):
    s1 = df[[id_col, standardise['Before']]].groupby(id_col).mean().reset_index()
    s1 = s1[s1[standardise['Before']] > 0]
    s1[x] = 'Before'
    s2 = df[[id_col, standardise['After']]].groupby(id_col).mean().reset_index()
    s2 = s2[s2[standardise['After']] > 0]
    s2[x] = 'After'

    if in_out_breakdown:
        df = df[[id_col] + [x] + [in_out_breakdown] + [hue] + [count_col]].groupby(
            [id_col] + [x] + [in_out_breakdown] + [hue]).count().reset_index()

    else:
        df = df[[id_col] + [x] + [hue] + [count_col]].groupby(
            [id_col] + [x] + [hue]).count().reset_index()

    df = df.merge(s1, on=[id_col, x], how='left')
    df = df.merge(s2, on=[id_col, x], how='left')
    df['norm'] = np.nan
    df['norm'][df[x] == 'Before'] = df[count_col] / df[standardise['Before']]
    df['norm'][df[x] == 'After'] = df[count_col] / df[standardise['After']]
    df.drop(count_col, axis=1, inplace=True)
    df[count_col] = df['norm']
    df.drop('norm', axis=1, inplace=True)
    return df



def PlotChi(aspect, bb, bw, ci, count_col, df_plot, estimator, figsize, hue, kind, legend, legend_loc, line, linestyle,
            save_name, save_path, title, title_fontsize, x, x_tick_fontsize, xlab, y_min, ylab, ylabs, yticks, pallette):
    sns.set_palette(pallette)
    plt.figure(figsize=figsize)
    if kind == 'violin':
        chart = sns.catplot(x=x, y=count_col, hue=hue, data=df_plot, kind=kind,
                            legend=False, ci=ci, bw=bw, cut=y_min, aspect=aspect)
    else:
        chart = sns.catplot(x=x, y=count_col, hue=hue, data=df_plot, kind=kind,
                            aspect=aspect, legend=False, ci=ci, estimator=estimator)
    chart.set_xticklabels(rotation=45, horizontalalignment='center')
    plt.xticks(fontsize=x_tick_fontsize)
    f = open(str(save_path) + str(save_name) + '.txt')
    text = f.read()
    chart.set(xlabel=xlab + "\n\n" +
                     text,
              ylabel=ylab)
    plt.axhline(y=line, color='grey', linestyle=linestyle)
    if y_min:
        plt.ylim(y_min)
    if yticks and ylabs:
        plt.yticks(ticks=yticks, labels=ylabs)
    if legend == 'within':
        chart.add_legend(loc=legend_loc)
    if legend == 'out':
        plt.legend(bbox_to_anchor=bb,
                   borderaxespad=0)
    plt.title(title, y=1, fontsize=title_fontsize)
    plt.subplots_adjust(right=0.2)
    plt.tight_layout()
    plt.savefig(save_path + save_name + ".png", bbox_inches='tight')
    plt.clf()
    plt.cla()
    plt.close()


def Twoway_ANOVA(df, group, x, y):
    t=[]
    #2way ANOVA
    equation = str(y) + " ~ " + str(x) + " + " + str(group) + " + " + str(x)+ ":" + str(group)
    model = ols(equation, data=df).fit()
    res = sm.stats.anova_lm(model, typ=2)
    res["PR(>F)"]=res["PR(>F)"].map('{:,.4f}'.format)
    res["sig"]=' '
    nrow = res.shape[0]
    for row in range(0, nrow):
        p = float(res["PR(>F)"][row])
        if p < 0.001:
            res["sig"][row] = '***'
        elif p < 0.01:
            res["sig"][row] = '**'
        elif p < 0.05:
            res["sig"][row] = '*'
        else:
            pass
    print(res)
    return t


def score_distribution(sample_scores, sample_scores_base, y_train, y_test, y):
    diff = [y - x for y, x in zip(sample_scores, sample_scores_base)]
    # Comopute the mean of differences
    d_bar = np.mean(diff)
    # compute the variance of differences
    sigma2 = np.var(diff)
    # compute the number of data points used for training
    n1 = len(y_train)
    # compute the number of data points used for testing
    n2 = len(y_test)
    # compute the total number of data points
    n0 = len(y)
    # compute the modified variance
    sigma2_mod = sigma2 * (1 / n0 + n2 / n1)
    # compute the t_static
    t_static = d_bar / np.sqrt(sigma2_mod)
    # Compute p-value
    Pvalue = ((1 - t.cdf(t_static, n0 - 1)) * 200)
    print(str(Pvalue))


