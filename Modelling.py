import os
import itertools
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import statsmodels.api as sm
from matplotlib import pyplot
from statsmodels.stats.contingency_tables import mcnemar
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, plot_confusion_matrix, precision_recall_curve
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.feature_selection import RFECV, SelectFromModel
from itertools import product

from Functions.Plotting import plot_precis_recall_curve, plot_precis_recall_curve_compare, plot_roc, plot_roc_compare

print('running')

perc = 50
Temeke = False
save_params = True
full_n_vars = 15
selected_n_vars = 9

save_param_path = "Results/Modelling/Grid_Search_Results/"
curve_plot_path = "Results/Modelling/Precision_recall_curves/"
roc_curve_plot_path = "Results/Modelling/ROC_curves/"

if Temeke == True:
    t = ""
if Temeke == False:
    t = "_No_Temeke"

# Load Data:
X_data_path = "Data/Modelling/Features/"
y_data_path = "Data/Modelling/DV/"

# load IVs
X_all = pd.read_csv(X_data_path + "Selected_Features_All_Reduced_By_IV_and_DVcors0.07.csv")

# check missing data:
percent_missing = round(X_all.isnull().sum() * 100 / len(X_all), 2)
missing_value_df = pd.DataFrame({'feature': X_all.columns,
                                 'percent_missing': percent_missing})
missing_value_df.sort_values(by='percent_missing', inplace=True, ascending=False, axis=0)
missing_value_df = missing_value_df[missing_value_df["percent_missing"]>0]
missing_value_df.to_csv("Data/Missing_Data/Missing_Modelling_Data0.07.csv")

# load DV
y = pd.read_csv(y_data_path + "Binary_Dependent_Variable_{}{}.csv".format(perc, t), index_col=None, usecols=["ID", "Binary_DV"])

# join
X_and_y = X_all.merge(y, how='inner', on="ID")
X_and_y.set_index("ID", inplace=True)
X_all = X_and_y.drop('Binary_DV', axis=1)
y = X_and_y['Binary_DV']

# ----------------------------------------------------------------------------------------------------------------------
print_path = "Results/Modelling/Print_Results/"

dt_params = {"classification__min_samples_split": [2, 3, 4],
             "classification__max_depth" : [2, 3, 4, 5],
             "classification__random_state": [1, 2, 3, 4, 5],
             "classification__max_features": ["sqrt", "log2"],
             "classification__class_weight": ["balanced", None]}
rf_params = {"classification__min_samples_split": [2, 3, 4, 5],
             "classification__max_depth" : [2, 3, 4, 5, 6],
             "classification__n_estimators": [10],
             "classification__random_state": [1, 2, 3, 4, 5],
             "classification__max_features": ["sqrt", "log2"],
             "classification__class_weight": ["balanced", None]}
log_params = {"classification__penalty" : ["l2","l1","elasticnet"],
              "classification__C" : [1, 1.5, 2, 5],
              "classification__tol" : [0.0000001, 0.0001],
              "classification__random_state": [1, 2, 3, 4, 5],
              "classification__solver":['newton-cg', 'lbfgs', 'liblinear', 'saga'],
              "classification__class_weight": ["balanced", None],
              "classification__max_iter": [100, 1000],
              "classification__l1_ratio": [0.25, 0.5, 0.75]}


penaltys = ["l2","l1","elasticnet"]
Cs = [0.5, 1, 1.25, 1.5, 2, 2.5, 3]
tols = [0.0000001, 0.0001]
random_states = [1, 2, 3, 4, 5]
solvers = ['newton-cg', 'lbfgs', 'liblinear', 'saga']
class_weights = ["balanced", None]
max_iters = [1000, 2000]
l1_ratios = [0.15, 0.25, 0.35, 0.5, 0.65, 0.75, 0.85]

all_params = list(product(penaltys, Cs, tols, random_states, solvers, class_weights, max_iters, l1_ratios))
filtered_log_params = [{'classification__penalty' : [penalty], 'classification__C': [C], 'classification__tol': [tol],
                        'classification__random_state': [random_state], 'classification__solver': [solver],
                        'classification__class_weight' : [class_weight], 'classification__max_iter': [max_iter],
                        'classification__l1_ratio' : [l1_ratio]}
                        for penalty, C, tol, random_state, solver, class_weight, max_iter, l1_ratio in all_params
                        if not (solver == 'newton-cg' and (penalty == 'l1' or penalty == 'elasticnet'))
                        and not (solver == 'lbfgs' and (penalty == 'l1' or penalty == 'elasticnet'))
                        and not (solver == 'liblinear' and penalty == 'elasticnet')
                        and not ((penalty == 'l1' or penalty == 'l2') and
                                 (l1_ratio == 0.25 or l1_ratio == 0.5 or l1_ratio == 0.75))]


best_log_params = {'classification__C': [1.5], 'classification__class_weight': ['balanced'], 'classification__l1_ratio': [0.15],
               'classification__max_iter': [1000], 'classification__penalty': ['l1'], 'classification__random_state': [1],
               'classification__solver': ['saga'], 'classification__tol': [1e-07]}

save_path = "Results/Modelling/"

model = []
score_list = []
params = []
error_array = []
matrix = []

cv = 10
verbose = 0
round_n = 5
count = 1
train_scoring = 'f1'
test_scoring = 'f1'
# todo: add in recall scoring and accuracy as well

start_time = dt.datetime.now()
dv = "Binary_DV"

results_save_file = "Results_{}{}_{}_{}.csv".format(test_scoring, t, perc, str(dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
print(str(dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")), file=open(str(print_path) + str(results_save_file), "w"))

# just before:
before_only = ['Before_Sent_Calls', 'Before_Call_Entropy', 'Before_Mode_Calling_Distance',
               'Before_Perc_of_Calls_to_Dar', 'Before_Calls_to_Dar_Ward_Rank', 'Before_Calls_to_Dar_Subward_Rank',
               'Before_Mean_Balance','Before_Money_Received_Normed', 'Before_Money_Sent_Normed',
               'Before_Cashin_Normed', 'Before_CashOut_Normed', 'Before_UtilityBill_Normed', 'Had_MFS_Before','Region_Arusha', 'Region_Dodoma', 'Region_Geita', 'Region_Iringa',
               'Region_Kagera', 'Region_Katavi', 'Region_Kigoma', 'Region_Kilimanjaro',
               'Region_Lindi', 'Region_Manyara', 'Region_Mara', 'Region_Mbeya',
               'Region_Morogoro', 'Region_Mtwara', 'Region_Mwanza', 'Region_Njombe',
               'Region_Pemba Island', 'Region_Pwani', 'Region_Rukwa', 'Region_Ruvuma',
               'Region_Shinyanga', 'Region_Simiyu', 'Region_Singida', 'Region_Tabora',
               'Region_Tanga', 'Region_Zanzibar',
               "Region_HDI", "Region_Life_Expectancy", "Region_Expected_Years_Schooling", "Region_GDPpc", "Region_NonIncome_HDI",
               "Region_Ratio_Female_Male_HDI", "Region_Life_Expectancy_Female", "Region_Life_Expectancy_Male", "Region_EYS_Female_Male_Ratio",
               "Region_Estimate_GDPpc_Female", "Region_Estimate_GDPpc_Male", "Region_Multidim_Poverty_Index", "Region_Incidence_Poverty",
               "Region_Av_Intensity_Poor", "Region_Population_Vulnerable_To_Poverty", "Region_Population_In_Severe_Poverty", "Region_Standard_of_Living",
               "Region_Health", "Region_Education", "Region_Immunization_DPT", "Region_Immunization_Measles", "Region_Child_Nutrition_Stunted",
               "Region_Child_Nutrition_Wasted", "Region_Under_Weight_Birth", "Region_Death_at_Delivery_Facility", "Region_Delivered_Skilled_Provider",
               "Region_Delivered_Facility", "Region_Antenatal_Visits", "Region_Women_HIV",	"Region_Men_HIV", "Region_Parliament_Female_Male_Ratio",
               "Region_District_Commissioners_Female_Male_Ratio", "Region_Concillors_Female_Male_Ratio", "Region_Adult_Literacy",
               "Region_Population_Secondary_Education", "Region_Population_Millions",
               "Region_Population_0_4years", "Region_Population_5_14years",	"Region_Population_15_64years",	"Region_Population_65years_plus",
               "Region_Urban_Population", "Region_Population_Density_km2", "Region_Median_Age", "Region_Age_Dependency_Ratio",
               "Region_Fertility_Rate",	"Region_Male_Female_Ratio", 'Visited_Dar_Before_Moved', 'Rural_Ward']

X_all = X_all[[c for c in X_all.columns if c in before_only]]

# Train and evaluate models
X_train, X_test, y_train, y_test = train_test_split(X_all, y, test_size=0.2, random_state=93, stratify=y)

# for log baseline
vars = ['Before_Calls_to_Dar_Ward_Rank', 'Before_Calls_to_Dar_Subward_Rank', 'Region_Immunization_Measles',
            'Region_Antenatal_Visits', "Region_Parliament_Female_Male_Ratio", "Before_Perc_of_Calls_to_Dar", "Region_Mara",
            "Region_Ruvuma", "Region_Shinyanga"]

X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(X_all[vars], y, test_size=0.2, random_state=93, stratify=y)

# pipeline steps:
imputer = IterativeImputer(max_iter=5, random_state=0)
transformer = StandardScaler()
log = LogisticRegression()
# selector = RFECV(log, step=1, min_features_to_select=8)
selector = RFECV(log, step=1, cv=10)

# ---- fit lm model (out of interest):
fit_imputer = IterativeImputer(max_iter=5, random_state=0).fit(X_all[vars])
fit_X = fit_imputer.transform(X_all[vars])
log_reg = sm.Logit(y, fit_X).fit()
print(log_reg.summary())

# get coefs
lm = LogisticRegression()
lm.fit(fit_X, y)
coef = pd.DataFrame(lm.coef_, index=X_all[vars_all].columns, columns=['coef']).sort_values(by='coef')
coef.to_csv(save_path + "Fit_best_model_lm_coef_{}{}.csv".format(perc, t))

# 0 Baseline
if count == 1:
    if test_scoring != 'f1':
        # a) Constant Negative -------------------------
        mean_baseline = DummyClassifier(strategy="constant",
                                        constant=0)
        mean_baseline.fit(X_train, y_train)
        y_pred = mean_baseline.predict(X_test)
        if test_scoring == 'recall':
            score = round(metrics.recall_score(y_test, y_pred), round_n)

        # record test results:
        model.append('Constant_Negative_Baseline')
        score_list.append(score)
        params.append('-')
        matrix.append('-')

        # print:
        print("Constant Negative Baseline: {} score = {}".format(test_scoring, score),
              file=open(str(print_path) + str(results_save_file), "a"))

    # b) Random Baseline ---------------------------
    rand_baseline = DummyClassifier(strategy="uniform")
    rand_baseline.fit(X_train, y_train)
    y_pred_rand = rand_baseline.predict(X_test)
    if test_scoring == 'f1':
        score = round(metrics.f1_score(y_test, y_pred_rand), round_n)
    elif test_scoring == 'recall':
        score = round(metrics.recall_score(y_test, y_pred_rand), round_n)

    # record test results:
    model.append('Random_Baseline')
    score_list.append(score)
    params.append('-')
    matrix.append('-')

    # print:
    print("Random Baseline: {} score = {}".format(test_scoring, score),
          file=open(str(print_path) + str(results_save_file), "a"))

    # c) Constant Positive --------------------------
    pos_baseline = DummyClassifier(strategy="constant",
                                    constant=1)
    pos_baseline.fit(X_train, y_train)
    y_pred_base = pos_baseline.predict(X_test)
    if test_scoring == 'f1':
        score = round(metrics.f1_score(y_test, y_pred_base), round_n)
    elif test_scoring == 'recall':
        score = round(metrics.recall_score(y_test, y_pred_base), round_n)

    err_base = []
    for actual, pred in zip(y_test_log, y_pred_base):
        if actual == pred:
            err_base.append('True')
        if actual != pred:
            err_base.append('False')

    # precision-recall curve for baseline
    yhat_base = pos_baseline.predict_proba(X_test)
    plot_precis_recall_curve(yhat=yhat_base, y=y, y_test=y_test, save_path=curve_plot_path,
                             model_lab="Constant Positive Baseline",
                             save_name="prec_recall_curve_constant_pos_baseline")

    # record test results:
    model.append('Constant_Positive_Baseline')
    score_list.append(score)
    params.append('-')
    matrix.append('-')

    # print:
    print("Constant Positive Baseline: {} score = {}".format(test_scoring, score),
              file=open(str(print_path) + str(results_save_file), "a"))

    # d) Positive correlation vars (>=0.8) in a regression (no optimisation) -----
    log_baseline = Pipeline([
        ('imputing', imputer),
        ('scaling', transformer),
        ('feature_selection', selector),
        ('classification', log)
    ])
    log_baseline.fit(X_train_log, y_train_log)
    y_pred = log_baseline.predict(X_test_log)

    if test_scoring == 'f1':
        score = round(metrics.f1_score(y_test_log, y_pred), round_n)
    elif test_scoring == 'recall':
        score = round(metrics.recall_score(y_test_log, y_pred), round_n)

    confusion_matrix = np.round(metrics.confusion_matrix(y_test_log, y_pred, normalize='all', labels=[0, 1]), 2)
    plot_confusion_matrix(estimator=log_baseline, X=X_test_log, y_true=y_test_log, normalize='all', labels=[0, 1], display_labels=["Not", "Vulnerable"])
    plt.savefig(save_path + "Confusion_Matrix_Baseline_Log_Regression_{}.png".format(str(dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))))

    # record test results:
    model.append('Log_Regression_Baseline')
    score_list.append(score)
    params.append(vars)
    matrix.append(confusion_matrix)

    # print:
    print("Log Regression Baseline: {} score = {}".format(test_scoring, score),
              file=open(str(print_path) + str(results_save_file), "a"))

# 1a) Logistic Regression (feature where r>=0.08) ----------------------------------
start_log = dt.datetime.now()
log = LogisticRegression()

# further feature selection:
selector = RFECV(log, step=1, cv=cv)

pipe = Pipeline([
    ('imputing', imputer),
    ('scaling', transformer),
    ('feature_selection', selector),
    ('classification', log)
])

# search best log_params
grid_search = GridSearchCV(estimator=pipe,
                           param_grid=filtered_log_params,
                           cv=cv,
                           scoring=train_scoring,
                           refit=False, verbose=verbose)

grid_search.fit(X_train_log, y_train_log)
best_params = grid_search.best_params_
best_score = round(abs(grid_search.best_score_), round_n)

# save param df:
if save_params == True:
    param_df = pd.DataFrame(grid_search.cv_results_).sort_values(by='rank_test_score')
    param_df.to_csv(save_param_path+"Logistic_Regression_{}_{}.csv".format(selected_n_vars, str(dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))))

# define best model and fit to training
pipe.set_params(**best_params)
pipe.fit(X_train_log, y_train_log)

# look at what features kept:
keep_df = pd.DataFrame(pipe.named_steps.feature_selection.support_, index=X_all[vars].columns)
keep_df.to_csv(save_path+"Feature_Selection_Logistic_Regression_{}_{}.csv".format(selected_n_vars, str(dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))))

# predict and score
y_pred = pipe.predict(X_test_log)
if test_scoring == 'f1':
    score = round(metrics.f1_score(y_test_log, y_pred), round_n)
elif test_scoring == 'recall':
    score = round(metrics.recall_score(y_test_log, y_pred), round_n)

err = []
for actual, pred in zip(y_test_log, y_pred):
    if actual == pred :
        err.append('True')
    if actual != pred:
        err.append('False')

confusion_matrix = np.round(metrics.confusion_matrix(y_test_log, y_pred, normalize='all', labels=[0, 1]), 2)
plot_confusion_matrix(estimator=pipe, X=X_test_log, y_true=y_test_log, normalize='all', labels=[0, 1], display_labels=["Not", "Vulnerable"])
plt.savefig(save_path + "Confusion_Matrix_Logistic_Regression_{}_{}.png".format(selected_n_vars, str(dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))))

# Significantly different than constant positive baseline?
crosstab = pd.crosstab(np.array(err), np.array(err_base), margins=False)
crosstab = crosstab[['True', 'False']]
crosstab.sort_index(ascending=False, inplace=True)
print('row= model, col= baseline : ', file=open(str(print_path) + str(results_save_file), "a"))
print(crosstab, file=open(str(print_path) + str(results_save_file), "a"))

result = mcnemar(crosstab, exact=True)
p = f'{result.pvalue:.4f}'
stat = f'{result.statistic:.4f}'
print("McNemar test of homogeneity: Stat={}; p={}".format(stat, p),
              file=open(str(print_path) + str(results_save_file), "a"))

# record test results:
model.append('Logistic_Regression{}'.format(selected_n_vars))
score_list.append(score)
params.append(best_params)
matrix.append(confusion_matrix)

# print:
print("Logistic Regression {} : {} score = {}".format(selected_n_vars, test_scoring, score),
              file=open(str(print_path) + str(results_save_file), "a"))
print("Best Params: {}".format(best_params),
              file=open(str(print_path) + str(results_save_file), "a"))
print('Logistic model 1 done. Time taken : ' + str(dt.datetime.now() - start_log),
              file=open(str(print_path) + str(results_save_file), "a"))

# -----------------------------------------------------------------------------------
# 1b) Logistic Regression (all features, r >=0.07) ----------------------------------
start_log = dt.datetime.now()
log = LogisticRegression()

pipe = Pipeline([
    ('imputing', imputer),
    ('scaling', transformer),
    ('feature_selection', selector),
    ('classification', log)
])

# search best log_params
grid_search = GridSearchCV(estimator=pipe,
                           param_grid=best_log_params,
                           cv=cv,
                           scoring=train_scoring,
                           refit=False, verbose=verbose)

grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
best_score = round(abs(grid_search.best_score_), round_n)

# save param df:
if save_params == True:
    param_df = pd.DataFrame(grid_search.cv_results_).sort_values(by='rank_test_score')
    param_df.to_csv(save_param_path+"Logistic_Regression_{}_{}.csv".format(full_n_vars, str(dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))))

# define best model and fit to training
pipe.set_params(**best_params)
pipe.fit(X_train, y_train)

# look at what features kept:
keep_df = pd.DataFrame(pipe.named_steps.feature_selection.support_, index=X_all.columns)
keep_df.to_csv(save_path+"Feature_Selection_Logistic_Regression_{}_{}.csv".format(full_n_vars, str(dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))))

# predict and score
y_pred = pipe.predict(X_test)
if test_scoring == 'f1':
    score = round(metrics.f1_score(y_test, y_pred), round_n)
elif test_scoring == 'recall':
    score = round(metrics.recall_score(y_test, y_pred), round_n)

err = []
for actual, pred in zip(y_test_log, y_pred):
    if actual == pred :
        err.append('True')
    if actual != pred:
        err.append('False')

confusion_matrix = np.round(metrics.confusion_matrix(y_test, y_pred, normalize='all', labels=[0, 1]), 2)
plot_confusion_matrix(estimator=pipe, X=X_test, y_true=y_test, normalize='all', labels=[0, 1], display_labels=["Not", "Vulnerable"])
plt.savefig(save_path + "Confusion_Matrix_Logistic_Regression_{}_{}.png".format(full_n_vars, str(dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))))

confusion_matrix2 = np.round(metrics.confusion_matrix(y_test, y_pred, normalize=None, labels=[0, 1]), 2)
plot_confusion_matrix(estimator=pipe, X=X_test, y_true=y_test, normalize=None, labels=[0, 1], display_labels=["Not", "Vulnerable"])
plt.savefig(save_path + "Confusion_Matrix_Logistic_Regression_{}_{}_NOTNORMED.png".format(full_n_vars, str(dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))))

# precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
yhat_model = pipe.predict_proba(X_test)
plot_precis_recall_curve(yhat=yhat_model, y=y, y_test=y_test, save_path=curve_plot_path,
                         save_name="prec_recall_curve_model_{}".format(str(dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))),
                         model_lab="Logistic")

plot_precis_recall_curve_compare(yhat1=yhat_model, yhat2=yhat_base, y_test=y_test, save_path=curve_plot_path,
                                 save_name="prec_recall_curve_compare_model_base_{}".format(str(dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))),
                                 model_lab1="Logistic Regression",
                                 model_lab2="Constant Positive Baseline", y=y)

# ROC curve
plot_roc(probs=yhat_model, y_test=y_test, save_path=roc_curve_plot_path,
         save_name="roc_curve_model_{}".format(str(dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))),
         model_name='Logistic')
plot_roc_compare(probs1=yhat_model, probs2=yhat_base, y_test=y_test, save_path=roc_curve_plot_path,
                 save_name="roc_curve_compare_model_base_{}".format(str(dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))),
                 model1_name="Logistic Regression",
                 model2_name="Constant Positive Baseline")

# Significantly different than constant positive baseline?
crosstab = pd.crosstab(np.array(err), np.array(err_base), margins=False)
crosstab = crosstab[['True', 'False']]
crosstab.sort_index(ascending=False, inplace=True)
print('row= model, col= baseline : ', file=open(str(print_path) + str(results_save_file), "a"))
print(crosstab,file=open(str(print_path) + str(results_save_file), "a"))

result = mcnemar(crosstab, exact=True)
p = f'{result.pvalue:.5f}'
stat = f'{result.statistic:.5f}'
print("McNemar test of homogeneity: Stat={}; p={}".format(stat, p),
              file=open(str(print_path) + str(results_save_file), "a"))

# record test results:
model.append('Logistic_Regression{}'.format(full_n_vars))
score_list.append(score)
params.append(best_params)
matrix.append(confusion_matrix)

# print:
print("Logistic Regression {} : {} score = {}".format(full_n_vars, test_scoring, score),
              file=open(str(print_path) + str(results_save_file), "a"))
print("Best Params: {}".format(best_params),
              file=open(str(print_path) + str(results_save_file), "a"))
print('Logistic model 2 done. Time taken : ' + str(dt.datetime.now() - start_log),
              file=open(str(print_path) + str(results_save_file), "a"))
breakpoint()
# ------------------------------------------------------------------------------------------------------------------
# 2 Decision Tree --------------------------------------------------------------------------------------------------
start_dt = dt.datetime.now()
tree = DecisionTreeClassifier()

pipe = Pipeline([
    ('imputing', imputer),
    ('scaling', transformer),
    ('classification', tree)
])

# search best dt_params
grid_search_dt = GridSearchCV(estimator=pipe, param_grid=dt_params, cv=cv,
                           scoring=train_scoring,
                           refit=False, verbose=verbose)
grid_search_dt.fit(X_train, y_train)
best_params = grid_search_dt.best_params_
best_score = round(abs(grid_search_dt.best_score_), round_n)

# save param df:
if save_params == True:
    param_df = pd.DataFrame(grid_search_dt.cv_results_).sort_values(by='rank_test_score')
    param_df.to_csv(save_param_path+"Decision_Tree_{}.csv".format(str(dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))))

# define best model and fit to training
pipe.set_params(**best_params)
pipe.fit(X_train, y_train)

# predict on test and score:
y_pred = pipe.predict(X_test)
if test_scoring == 'f1':
    score = round(metrics.f1_score(y_test, y_pred), round_n)
elif test_scoring == 'recall':
    score = round(metrics.recall_score(y_test, y_pred), round_n)
confusion_matrix = np.round(metrics.confusion_matrix(y_test, y_pred, normalize='all', labels=[0, 1]), 2)
plot_confusion_matrix(estimator=pipe, X=X_test, y_true=y_test, normalize='all', labels=[0, 1], display_labels=["Not", "Vulnerable"])
plt.savefig(save_path + "Confusion_Matrix_Decision_Tree_{}.png".format(str(dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))))

# record test results:
model.append('Decision_Tree')
score_list.append(score)
params.append(best_params)
matrix.append(confusion_matrix)

print("Decision Tree: {} score = {}".format(test_scoring, score),
              file=open(str(print_path) + str(results_save_file), "a"))
print("Best Params: {}".format(best_params),
              file=open(str(print_path) + str(results_save_file), "a"))
print('Decision tree done. Time taken : ' + str(dt.datetime.now() - start_dt),
              file=open(str(print_path) + str(results_save_file), "a"))

# 3) Random Forest --------------------------------------
start_rf = dt.datetime.now()
rf = RandomForestClassifier()

pipe = Pipeline([
    ('imputing', imputer),
    ('scaling', transformer),
    ('classification', rf)
])

# search best dt_params
grid_search_rf = GridSearchCV(estimator=pipe, param_grid=rf_params, cv=cv,
                           scoring=train_scoring,
                           refit=False, verbose=verbose)
grid_search_rf.fit(X_train, y_train)
best_params = grid_search_rf.best_params_
best_score = round(abs(grid_search_rf.best_score_), round_n)

# save param df:
if save_params == True:
    param_df = pd.DataFrame(grid_search_rf.cv_results_).sort_values(by='rank_test_score')
    param_df.to_csv(save_param_path+"Random_Forest_{}.csv".format(str(dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))))

# define best model and fit to training
pipe.set_params(**best_params)
pipe.fit(X_train, y_train)

# predict on test and score:
y_pred = pipe.predict(X_test)
if test_scoring == 'f1':
    score = round(metrics.f1_score(y_test, y_pred), round_n)
elif test_scoring == 'recall':
    score = round(metrics.recall_score(y_test, y_pred), round_n)
confusion_matrix = np.round(metrics.confusion_matrix(y_test, y_pred, normalize='all', labels=[0, 1]), 2)
plot_confusion_matrix(estimator=pipe, X=X_test, y_true=y_test, normalize='all', labels=[0, 1], display_labels=["Not", "Vulnerable"])
plt.savefig(save_path + "Confusion_Matrix_Random_Forest_{}.png".format(str(dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))))

# record test results:
model.append('Random_Forest')
score_list.append(score)
params.append(best_params)
matrix.append(confusion_matrix)

print("Random Forest: {} score = {}".format(test_scoring, score),
              file=open(str(print_path) + str(results_save_file), "a"))
print("Best Params: {}".format(best_params),
              file=open(str(print_path) + str(results_save_file), "a"))
print('Random forest done. Time taken : ' + str(dt.datetime.now() - start_dt),
              file=open(str(print_path) + str(results_save_file), "a"))
# --------------------------------------------------------------------------------------------------------------------
print("Total time taken : " + str(dt.datetime.now() - start_time),
              file=open(str(print_path) + str(results_save_file), "a"))

# Sort results by accuracy
list_dict = {'Model': model, "Score": score_list, "Best_Params": params, "Confusion_Matrix": matrix}
results = pd.DataFrame.from_dict(list_dict)
results.sort_values(by='Score', inplace=True, ascending=False)
results = results.reset_index()
results.to_csv(save_path + "Results_{}_{}.csv".format(test_scoring, str(dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))))


print("done!")