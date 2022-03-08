import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.contingency_tables import mcnemar
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, plot_confusion_matrix
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.feature_selection import RFECV, SelectFromModel, RFE
from itertools import product

drop = False
perc = 50
Temeke = False

if drop == True:
    d = "_Drop"
    nfeatures = 7
if drop == False:
    d = ""
    nfeatures = 8

if Temeke == True:
    t = ""
if Temeke == False:
    t = "_No_Temeke"

# Load Data:
X_data_path = "Data/Modelling/Features/"
y_data_path = "Data/Modelling/DV/"

# load IVs
X_all = pd.read_csv(X_data_path + "Selected_Features_All_Reduced_By_IV_and_DVcors0.07.csv")
X_all.set_index("ID", inplace=True)
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

# load DV
y = pd.read_csv(y_data_path + "Binary_Dependent_Variable_{}{}.csv".format(perc, t), index_col=None, usecols=["ID", "Binary_DV"])
y.set_index("ID", inplace=True)

# join
X_and_y = X_all.merge(y, how='inner', left_index=True, right_index=True)
X_and_y.to_csv("Data/Modelling/X_and_y_Final/X_and_y{}{}.csv".format(perc, t))
X_all = X_and_y.drop('Binary_DV', axis=1)
y = X_and_y['Binary_DV']

best_params = {'classification__C': 1.5, 'classification__class_weight': 'balanced', 'classification__l1_ratio': 0.15,
               'classification__max_iter': 1000, 'classification__penalty': 'l1', 'classification__random_state': 1,
               'classification__solver': 'saga', 'classification__tol': 1e-07}


# -------------------------- fit best model:

# pipeline steps:
imputer = IterativeImputer(max_iter=5, random_state=0)
transformer = StandardScaler()
log = LogisticRegression()
selector = RFECV(log, step=1, cv=10)

pipe = Pipeline([
    ('imputing', imputer),
    ('scaling', transformer),
    ('feature_selection', selector),
    ('classification', log)
])

if drop == True:
    X_all.drop('Before_Calls_to_Dar_Ward_Rank', inplace=True, axis=1)

pipe.set_params(**best_params)
pipe.fit(X_all, y)

coefs = round(pd.DataFrame(pipe.named_steps.classification.coef_[0], columns=["Coef"]), 2)
ranks = list(pipe.named_steps.feature_selection.ranking_)
vars = X_all.columns.to_list()

dict = {'Feature': vars, "Rank": ranks}
rank_df = pd.DataFrame(dict)

selected_df = rank_df[rank_df["Rank"]==1].reset_index()
fit_results_df = pd.concat([selected_df, coefs], axis=1)
fit_results_df = fit_results_df[["Feature", "Coef"]]

fit_save_path = "Results/Modelling/Best_Model/"
fit_results_df.to_csv(fit_save_path + "Fit_best_model_{}{}{}_{}.csv".format(perc, t, d, str(dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))))

print('done!')