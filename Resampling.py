import numpy as np
import pandas as pd
import datetime as dt
from sklearn import metrics
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import cross_val_predict, cross_val_score, train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE, RFECV
from Functions.Plotting import plot_hist_stacked
from Functions.Stat_tests import score_distribution
from sklearn.metrics import precision_recall_curve

stratify_base = True
n = 10000
round_n=3
score = 'AUC'

if stratify_base == False:
    strat = 'No_Base_Strat'
if stratify_base == True:
    strat = 'Base_Strat'

# get data
X_and_y = pd.read_csv("Data/Modelling/X_and_y_Final/X_and_y50_No_Temeke.csv", index_col=False)
X_and_y.set_index("ID", inplace=True)
X = X_and_y.drop('Binary_DV', axis=1)
y = X_and_y['Binary_DV']

# pipeline steps:
imputer = IterativeImputer(max_iter=5, random_state=0)
transformer = StandardScaler()
log = LogisticRegression()
# as testing best model can only use features selected previously:
selected_vars = ["Before_Call_Entropy",
"Before_Perc_of_Calls_to_Dar",
"Before_Calls_to_Dar_Ward_Rank",
"Before_Calls_to_Dar_Subward_Rank",
"Before_Money_Sent_Normed",
"Region_Education",
"Region_Manyara",
"Region_Mara",
"Region_Pwani",
"Region_Ruvuma",
"Region_Shinyanga"]

selector = RFECV(log, step=1, cv=10)

# def select(X, selected_vars):
#     X = X[selected_vars]
#     return X
#
# selector = select(X=X, selected_vars=selected_vars)

pipe = Pipeline([
    ('imputing', imputer),
    ('scaling', transformer),
    ('feature_selection', selector),
    ('classification', log)
])

best_grid = {'classification__C': [1.5],
               'classification__class_weight': ['balanced'],
               'classification__max_iter': [1000],
               'classification__penalty': ['l1'],
               'classification__random_state': [1],
               'classification__solver': ['saga'],
               'classification__tol': [1e-07]}

best_params = {'classification__C': 1.5,
               'classification__class_weight': 'balanced',
               'classification__max_iter': 1000,
               'classification__penalty': 'l1',
               'classification__random_state': 1,
               'classification__solver': 'saga',
               'classification__tol': 1e-07}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=93, stratify=y)

grid_search = GridSearchCV(estimator=pipe,
                           param_grid=best_grid,
                           cv=10,
                           scoring="f1",
                           refit=False, verbose=0)

# refit best model:
grid_search.fit(X_train, y_train)
pipe.set_params(**best_params)
pipe.fit(X_train, y_train)

sample_scores = []
sample_predictions = []

sample_scores_base = []
sample_predictions_base = []

sample_scores_rand_base_s = []
sample_predictions_rand_base_s = []

sample_scores_rand_base_u = []
sample_predictions_rand_base_u = []

for seed in range(0,n):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

    # MODEL ----------------------------
    # fit model:
    pipe.fit(X_train, y_train)

    # predict:
    y_pred_model = pipe.predict(X_test)

    # score:
    if score == 'F1':
        score_model = round(metrics.f1_score(y_test, y_pred_model), round_n)
    if score == "AUC":
        probs = pipe.predict_proba(X_test)
        probs = probs[:, 1]
        score_model = round(metrics.roc_auc_score(y_test, probs), round_n)

    sample_pred_df = pd.DataFrame(y_pred_model)
    sample_pred_df.index = y_test.index

    # Save to lists:
    sample_scores.append(score_model)
    sample_predictions.append(sample_pred_df)

    # CONSTANT POSITIVE BASE ------------
    if stratify_base == False:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    if stratify_base == True:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

    # fit model:
    constant_pos_baseline = DummyClassifier(strategy="constant", constant=1)
    constant_pos_baseline.fit(X_train, y_train)

    # predict:
    y_pred_constant_base = constant_pos_baseline.predict(X_test)

    # score:
    if score == 'F1':
        score_constant_pos_base = round(metrics.f1_score(y_test, y_pred_constant_base), round_n)
    if score == "AUC":
        probs = constant_pos_baseline.predict_proba(X_test)
        probs = probs[:, 1]
        score_constant_pos_base = round(metrics.roc_auc_score(y_test, probs), round_n)

    sample_pred_base_df = pd.DataFrame(y_pred_constant_base)
    sample_pred_base_df.index = y_test.index

    # Save to lists:
    sample_scores_base.append(score_constant_pos_base)
    sample_predictions_base.append(sample_pred_base_df)

    # RANDOM BASE STRATIFIED --------------
    if stratify_base == False:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    if stratify_base == True:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

    rand_s_baseline = DummyClassifier(strategy="stratified", random_state=seed)

    # fit model:
    rand_s_baseline.fit(X_train, y_train)

    # predict:
    y_pred_rand_base_s = rand_s_baseline.predict(X_test)

    # score:
    if score == 'F1':
        score_rand_base_s = round(metrics.f1_score(y_test, y_pred_rand_base_s), round_n)
    if score == "AUC":
        probs = rand_s_baseline.predict_proba(X_test)
        probs = probs[:, 1]
        score_rand_base_s = round(metrics.roc_auc_score(y_test, probs), round_n)

    sample_pred_rand_base_df = pd.DataFrame(y_pred_rand_base_s)
    sample_pred_rand_base_df.index = y_test.index

    # Save to lists:
    sample_scores_rand_base_s.append(score_rand_base_s)
    sample_predictions_rand_base_s.append(sample_pred_rand_base_df)

    # RANDOM BASE UNIFORM -----------------------
    if stratify_base == False:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    if stratify_base == True:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

    rand_u_baseline = DummyClassifier(strategy="uniform", random_state=seed)

    # fit model:
    rand_u_baseline.fit(X_train, y_train)

    # predict:
    y_pred_rand_base_u = rand_u_baseline.predict(X_test)

    # score:
    if score == 'F1':
        score_rand_base_u = round(metrics.f1_score(y_test, y_pred_rand_base_u), round_n)
    if score == "AUC":
        probs = rand_u_baseline.predict_proba(X_test)
        probs = probs[:, 1]
        score_rand_base_u = round(metrics.roc_auc_score(y_test, probs), round_n)

    sample_pred_rand_base_u_df = pd.DataFrame(y_pred_rand_base_u)
    sample_pred_rand_base_u_df.index = y_test.index

    # Save to lists:
    sample_scores_rand_base_u.append(score_rand_base_u)
    sample_predictions_rand_base_u.append(sample_pred_rand_base_u_df)

# join all datasets together:
save_path = "Data/Bootstrapped_Predictions/F1_Scores/"
model_scores = pd.DataFrame(sample_scores)
model_scores.to_csv(save_path + "Bootstrapped_Predictions_Model_n{}_{}.csv".format(n, str(dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))))

base_scores = pd.DataFrame(sample_scores_base)
base_scores.to_csv(save_path + "Bootstrapped_Predictions_Base_n{}_{}_{}.csv".format(n, score, str(dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))))

rand_base_scores = pd.DataFrame(sample_scores_rand_base_s)
rand_base_scores.to_csv(save_path + "Bootstrapped_Predictions_Base_n{}_{}_{}.csv".format(n, score, str(dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))))

rand_base_u_scores = pd.DataFrame(sample_scores_rand_base_u)
rand_base_u_scores.to_csv(save_path + "Bootstrapped_Predictions_Base_n{}_{}_{}.csv".format(n, score, str(dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))))

# --------------------
# constant positive baseline:
save_path = "Results/Bootstrapping/"
plot_hist_stacked(save_name="{}_Constant_Positive_Distributions_n{}{}{}".format(score, n, strat, str(dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))),
                  x1=base_scores,
                  x2=model_scores, bins=20,
                  save_path=save_path, x1_lab='Baseline', x2_lab='Model',
                  title="{} Distributions for n={} Bootstrapped Samples".format(score, n), title_fontsize = 14,
                  xlab="{} Score".format(score), ylab="Frequency", xlim=None, fig_size=(6,6))

# compute t-test on score distributions:
print('Positive Constant Baseline:')
score_distribution(sample_scores=sample_scores, sample_scores_base=sample_scores_base,
                   y_train=y_train, y_test=y_test, y=y)

# --------------------
# random base strat:
plot_hist_stacked(save_name="{}_RandomStrat_Base_Distributions_n{}{}{}".format(score, n, strat,str(dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))), x1=rand_base_scores,
                  x2=model_scores, bins=20,
                  save_path=save_path, x1_lab='Baseline', x2_lab='Model',
                  title="{} Distributions for n={} Bootstrapped Samples".format(score, n), title_fontsize = 14,
                  xlab="{} Score".format(score), ylab="Frequency", xlim=None, fig_size=(6,6))

# compute t-test on score distributions:
print('Stratified Random Baseline:')
score_distribution(sample_scores=sample_scores, sample_scores_base=sample_scores_rand_base_s,
                   y_train=y_train, y_test=y_test, y=y)

# --------------------
# random base uniform:
plot_hist_stacked(save_name="{}_RandomUni_Base_Distributions_n{}{}{}".format(score, n, strat,str(dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))), x1=rand_base_u_scores,
                  x2=model_scores, bins=20,
                  save_path=save_path, x1_lab='Baseline', x2_lab='Model',
                  title="{} Distributions for n={} Bootstrapped Samples".format(score, n), title_fontsize = 14,
                  xlab="{} Score".format(score), ylab="Frequency", xlim=None, fig_size=(6,6))

print('Uniform Random Baseline:')
# compute t-test on score distributions:
score_distribution(sample_scores=sample_scores, sample_scores_base=sample_scores_rand_base_u,
                   y_train=y_train, y_test=y_test, y=y)




print('done!')