# Load libraries.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import time
import datetime as dt
import re
import plotly.express as px
import pandas_datareader.data as web
from pandas_datareader import data as pdr
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

# Setting baseline seed
np.random.seed(241001)

# Set print options.
np.set_printoptions(precision = 3)
plt.style.use("ggplot") # Grammar of Graphics Theme
mpl.rcParams["axes.grid"] = True
mpl.rcParams["grid.color"] = "grey"
mpl.rcParams["grid.alpha"] = 0.25
mpl.rcParams["axes.facecolor"] = "white"
mpl.rcParams["legend.fontsize"] = 14
%matplotlib inline

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import chi2, f_regression
from sklearn.feature_selection import SelectKBest
import statsmodels.tsa.arima.model as stats
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from statsmodels.graphics.tsaplots import plot_acf

# Data Importing/Wrangling and Preparing

ff = pd.read_csv('FF_daily_data.csv')
ff = ff.rename(columns = {'Unnamed: 0':'Date'})
ff = ff.set_index('Date')
ff = ff.dropna()
ff = ff[['Mkt-RF','SMB','HML']]
ff.index = pd.to_datetime(ff.index)
stock_ticker_y = ["^RUA"]
stock_data_y = yf.download(stock_ticker_y)
stock_data_y = stock_data_y[['Adj Close']]
stock_data_y = stock_data_y.rename(columns ={'Adj Close':'RUA'})
stock_data_y = stock_data_y.loc['1999':'2023']
stock_data_y['RUA_return'] = np.exp(np.log(stock_data_y['RUA']/stock_data_y['RUA'].shift(1)))
stock_data_clean = pd.read_csv('data_cleaned_FINALNOW.csv')
stock_data_clean = stock_data_clean.set_index('Date')
return_data_clean= pd.DataFrame()
for i in list_of_col:
    return_data_clean[i+'_return']= np.exp(np.log(stock_data_clean[i] / stock_data_clean[i].shift(1)))
return_period = 5




# Outcome Variable (Y)

Y = np.log(stock_data_y.loc[:, ("RUA_return")]).diff(return_period).shift(-return_period)
Y.name = ('RUA' + "_pred")




# Independent Variables (X)

X1 = np.log(return_data_clean).diff(return_period)
X1.index = pd.to_datetime(X1.index)

X2 = pd.concat([np.log(stock_data_y.loc[:, "RUA_return"]).diff(i) for i in [return_period, return_period * 3, return_period * 6, return_period * 12]], axis = 1).dropna()
X2.columns = ["RUA_DT", "RUA_3DT", "RUA_6DT", "RUA_12DT"]

X3 = ff.diff(return_period)

dummydf = stock_data_clean.copy()

# Set GFC to 1 for the specified date range and 0 otherwise

dummydf['GFC'] = 0  # Initialize to 0
dummydf.loc[(return_data_clean.index >= '2007-08-09') & (return_data_clean.index <= '2008-12-31'), 'GFC'] = 1



# Set COVID to 1 for the specified date range and 0 otherwise

dummydf['COVID'] = 0  # Initialize to 0
dummydf.loc[(return_data_clean.index >= '2020-03-11') & (return_data_clean.index <= '2021-12-31'), 'COVID'] = 1



dummydf = dummydf[['GFC','COVID']]
dummydf[dummydf['COVID']!=0]
X4 = dummydf
X4.index = pd.to_datetime(X4.index)
X4.value_counts()

X5 = pd.concat([np.log(stock_data_y.loc[:, "RUA_return"]).rolling(i).mean() for i in [return_period, return_period * 3, return_period * 6, return_period * 12]], axis = 1).dropna()
X5.columns = ["RUA_MA_5", "RUA_MA_15", "RUA_MA_30", "RUA_MA_60"]

invar_stock_return = pd.read_csv('tensor_data2.csv')
invar_stock_return['datetime'] = pd.to_datetime(invar_stock_return['datetime'])
invar_stock_return = invar_stock_return.set_index('datetime')
X6 = invar_stock_return

constant_df = pd.DataFrame()
constant_df['constant'] = 1
date_range = pd.date_range(start='1999-01-01', end='2023-12-31', freq='D')
constant_df = constant_df.reindex(date_range)
constant_df.index.name = 'Date'
constant_df['constant'].fillna(1, inplace=True)

X = pd.concat([X1, X2, X3, X4, X5, X6, constant_df], axis = 1)
X.index = pd.to_datetime(X.index)

Y = Y.to_frame()
duplicate_indices = Y.index.duplicated()
duplicate_indices.sum()

X = X[~X.index.duplicated()]

data = pd.concat([Y, X], axis = 1).dropna().iloc[::return_period, :]
data.to_csv('data_combined_y_x.csv')

Y = data.loc[ : ,'RUA_pred']
X = data.loc[ : , X.columns]



# Data Split for training and testing

validation_size = 0.20
train_size = int(len(X) * (1 - validation_size))
X_train, X_test = (X[0  : train_size] , X[train_size:len(X)])
Y_train, Y_test = (Y[0  : train_size] , Y[train_size:len(X)])



# Feature Extraction

scaler = StandardScaler().fit(X_train)
rescaledDataset = pd.DataFrame(scaler.fit_transform(X_train),columns = X_train.columns, index = X_train.index)



# summarize transformed data

X_train.dropna(how = "any", inplace = True)
rescaledDataset.dropna(how = "any", inplace = True)
rescaledDataset.head(2)

scaler2 = StandardScaler().fit(X_test)
rescaledDataset2 = pd.DataFrame(scaler2.fit_transform(X_test),columns = X_test.columns, index = X_test.index)



# summarize transformed data

X_test.dropna(how = "any", inplace = True)
rescaledDataset2.dropna(how = "any", inplace = True)
rescaledDataset2.head(2)

from matplotlib.ticker import MaxNLocator
from sklearn.decomposition import TruncatedSVD

ncomps = 2
svd = TruncatedSVD(n_components=ncomps)
svd_fit = svd.fit(rescaledDataset)
plt_data = pd.DataFrame(svd_fit.explained_variance_ratio_.cumsum()*100)
plt_data.index = np.arange(1, len(plt_data) + 1)
Y_pred = svd.fit_transform(rescaledDataset)
ax = plt_data.plot(kind="line", figsize=(20, 10), style = "o-")
ax.xaxis.set_major_locator(MaxNLocator(integer = True))
ax.set_xlabel("PCs")
ax.set_ylabel("Percentage Explained")
ax.legend("")
print("Variance preserved by first 5 components == {:.2%}".format(svd_fit.explained_variance_ratio_.cumsum()[-1]))

dfsvd = pd.DataFrame(Y_pred, columns=["pc{}".format(c) for c in range(ncomps)], index = rescaledDataset.index)
print(dfsvd.shape)
dfsvd.tail()
X_train = dfsvd

ncomps = 2
svd = TruncatedSVD(n_components=ncomps)
svd_fit2 = svd.fit(rescaledDataset2)
plt_data2 = pd.DataFrame(svd_fit2.explained_variance_ratio_.cumsum()*100)
plt_data2.index = np.arange(1, len(plt_data) + 1)
Y_pred2 = svd.fit_transform(rescaledDataset2)
ax = plt_data2.plot(kind="line", figsize=(20, 10), style = "o-")
ax.xaxis.set_major_locator(MaxNLocator(integer = True))
ax.set_xlabel("PCs")
ax.set_ylabel("Percentage Explained")
ax.legend("")
print("Variance preserved by first 5 components == {:.2%}".format(svd_fit2.explained_variance_ratio_.cumsum()[-1]))

dfsvd2 = pd.DataFrame(Y_pred2, columns=["pc{}".format(c) for c in range(ncomps)], index = rescaledDataset2.index)
print(dfsvd2.shape)
dfsvd2.tail()
X_test = dfsvd2




# Finding the right model fit:

models = []

models.append(("LR", LinearRegression()))
models.append(("LASSO", Lasso()))
models.append(("EN", ElasticNet()))
models.append(("CART", DecisionTreeRegressor()))
models.append(("KNN", KNeighborsRegressor()))
models.append(("SVR", SVR()))
models.append(("RFR", RandomForestRegressor()))
models.append(("ETR", ExtraTreesRegressor()))
models.append(("GBR", GradientBoostingRegressor()))
models.append(("ABR", AdaBoostRegressor()))

num_folds = 10 
seed = 241001
scoring = "neg_mean_squared_error"

names = []
kfold_results = []
train_results = []
test_results = []

# Linear Regression
name = 'LinearRegression()'
model = LinearRegression()
kfold = KFold(n_splits = num_folds,
               random_state = seed,
               shuffle = True)
        
names.append(name)
cv_results = -1 * cross_val_score(model, 
                                  X_train, 
                                  Y_train,
                                  cv = kfold,
                                  scoring = scoring)
           
    
kfold_results.append(cv_results)
  
res_LN = model.fit(X_train, Y_train) # The model is trained on the entire training dataset.
train_result = mean_squared_error(res_LN.predict(X_train), Y_train)
train_results.append(train_result)
test_result = mean_squared_error(res_LN.predict(X_test), Y_test)
test_results.append(test_result)
print(name)
print('Mean Cross Validation Score: ', cv_results.mean())
print('Std Cross Validation Score: ', cv_results.std())
print('Train MSE: train_result: ', train_result)
print('Test MSE: test_result: ', test_result)

# Lasso

name = 'Lasso()'
model = Lasso()
kfold = KFold(n_splits = num_folds,
               random_state = seed,
               shuffle = True)
        
names.append(name)
cv_results = -1 * cross_val_score(model, 
                                  X_train, 
                                  Y_train,
                                  cv = kfold,
                                  scoring = scoring)
           
    
kfold_results.append(cv_results)
res_LS = model.fit(X_train, Y_train) # The model is trained on the entire training dataset.
train_result = mean_squared_error(res_LS.predict(X_train), Y_train)
train_results.append(train_result)
test_result = mean_squared_error(res_LS.predict(X_test), Y_test)
test_results.append(test_result)
print(name)
print('Mean Cross Validation Score: ', cv_results.mean())
print('Std Cross Validation Score: ', cv_results.std())
print('Train MSE: train_result: ', train_result)
print('Test MSE: test_result: ', test_result)


# Elastic Net

name = 'ElasticNet()'
model = ElasticNet()
kfold = KFold(n_splits = num_folds,
               random_state = seed,
               shuffle = True)
        
names.append(name)
cv_results = -1 * cross_val_score(model, 
                                  X_train, 
                                  Y_train,
                                  cv = kfold,
                                  scoring = scoring)
           
    
kfold_results.append(cv_results)
res_EN = model.fit(X_train, Y_train) # The model is trained on the entire training dataset
train_result = mean_squared_error(res_EN.predict(X_train), Y_train)
train_results.append(train_result)
test_result = mean_squared_error(res_EN.predict(X_test), Y_test)
test_results.append(test_result)
print(name)
print('Mean Cross Validation Score: ', cv_results.mean())
print('Std Cross Validation Score: ', cv_results.std())
print('Train MSE: train_result: ', train_result)
print('Test MSE: test_result: ', test_result)


# Decision Tree Regressor

name = 'DecisionTreeRegressor()'
model = DecisionTreeRegressor()
kfold = KFold(n_splits = num_folds,
               random_state = seed,
               shuffle = True)
        
names.append(name)
cv_results = -1 * cross_val_score(model, 
                                  X_train, 
                                  Y_train,
                                  cv = kfold,
                                  scoring = scoring)
           
kfold_results.append(cv_results)
res_DTR = model.fit(X_train, Y_train) # The model is trained on the entire training dataset.
train_result = mean_squared_error(res_DTR.predict(X_train), Y_train)
train_results.append(train_result)
test_result = mean_squared_error(res_DTR.predict(X_test), Y_test)  
test_results.append(test_result)
print(name)
print('Mean Cross Validation Score: ', cv_results.mean())
print('Std Cross Validation Score: ', cv_results.std())
print('Train MSE: train_result: ', train_result)
print('Test MSE: test_result: ', test_result)


# K Neighbors Regressor

name = 'KNeighborsRegressor()'
model = KNeighborsRegressor()
kfold = KFold(n_splits = num_folds,
               random_state = seed,
               shuffle = True)
        
names.append(name)
cv_results = -1 * cross_val_score(model, 
                                  X_train, 
                                  Y_train,
                                  cv = kfold,
                                  scoring = scoring)
           
   
kfold_results.append(cv_results)
res_KNR = model.fit(X_train, Y_train) # The model is trained on the entire training dataset
train_result = mean_squared_error(res_KNR.predict(X_train), Y_train)
train_results.append(train_result)
test_result = mean_squared_error(res_KNR.predict(X_test), Y_test)
test_results.append(test_result)
print(name)
print('Mean Cross Validation Score: ', cv_results.mean())
print('Std Cross Validation Score: ', cv_results.std())
print('Train MSE: train_result: ', train_result)
print('Test MSE: test_result: ', test_result)

# SVR

name = 'SVR()'
model = SVR()
kfold = KFold(n_splits = num_folds,
               random_state = seed,
               shuffle = True)
        
names.append(name)
cv_results = -1 * cross_val_score(model, 
                                  X_train, 
                                  Y_train,
                                  cv = kfold,
                                  scoring = scoring)
           
    
kfold_results.append(cv_results)
res_SVR = model.fit(X_train, Y_train) # The model is trained on the entire training dataset.
train_result = mean_squared_error(res_SVR.predict(X_train), Y_train)
train_results.append(train_result)
test_result = mean_squared_error(res_SVR.predict(X_test), Y_test)
test_results.append(test_result)
print(name)
print('Mean Cross Validation Score: ', cv_results.mean())
print('Std Cross Validation Score: ', cv_results.std())
print('Train MSE: train_result: ', train_result)
print('Test MSE: test_result: ', test_result)

# Random Forest Regressor

name = 'RandomForestRegressor()'
model = RandomForestRegressor()
kfold = KFold(n_splits = num_folds,
               random_state = seed,
               shuffle = True)
        
names.append(name)
cv_results = -1 * cross_val_score(model, 
                                  X_train, 
                                  Y_train,
                                  cv = kfold,
                                  scoring = scoring)
           
    
kfold_results.append(cv_results)
res_RFR = model.fit(X_train, Y_train) # The model is trained on the entire training dataset.
train_result = mean_squared_error(res_RFR.predict(X_train), Y_train)
train_results.append(train_result)
test_result = mean_squared_error(res_RFR.predict(X_test), Y_test)
test_results.append(test_result)
print(name)
print('Mean Cross Validation Score: ', cv_results.mean())
print('Std Cross Validation Score: ', cv_results.std())
print('Train MSE: train_result: ', train_result)
print('Test MSE: test_result: ', test_result)


# Extra Trees Regressor

name = 'RandomForestRegressor()'
model = RandomForestRegressor()
kfold = KFold(n_splits = num_folds,
               random_state = seed,
               shuffle = True)
        
names.append(name)
cv_results = -1 * cross_val_score(model, 
                                  X_train, 
                                  Y_train,
                                  cv = kfold,
                                  scoring = scoring)
           
    
kfold_results.append(cv_results)
res_ETR = model.fit(X_train, Y_train) # The model is trained on the entire training dataset.
train_result = mean_squared_error(res_ETR.predict(X_train), Y_train)
train_results.append(train_result)
test_result = mean_squared_error(res_ETR.predict(X_test), Y_test)
test_results.append(test_result)
print(name)
print('Mean Cross Validation Score: ', cv_results.mean())
print('Std Cross Validation Score: ', cv_results.std())
print('Train MSE: train_result: ', train_result)
print('Test MSE: test_result: ', test_result)


# Gradient Boosting Regressor

name = 'RandomForestRegressor()'
model = RandomForestRegressor()
kfold = KFold(n_splits = num_folds,
               random_state = seed,
               shuffle = True)
        
names.append(name)
cv_results = -1 * cross_val_score(model, 
                                  X_train, 
                                  Y_train,
                                  cv = kfold,
                                  scoring = scoring)
           
    
kfold_results.append(cv_results)
res_GBR = model.fit(X_train, Y_train) # The model is trained on the entire training dataset.
train_result = mean_squared_error(res_GBR.predict(X_train), Y_train)
train_results.append(train_result)
test_result = mean_squared_error(res_GBR.predict(X_test), Y_test)
test_results.append(test_result)
print(name)
print('Mean Cross Validation Score: ', cv_results.mean())
print('Std Cross Validation Score: ', cv_results.std())
print('Train MSE: train_result: ', train_result)
print('Test MSE: test_result: ', test_result)



# Ada Boost Regressor

name = 'RandomForestRegressor()'
model = RandomForestRegressor()
kfold = KFold(n_splits = num_folds,
               random_state = seed,
               shuffle = True)
        
names.append(name)
cv_results = -1 * cross_val_score(model, 
                                  X_train, 
                                  Y_train,
                                  cv = kfold,
                                  scoring = scoring)
           
    
kfold_results.append(cv_results)
res_ABR = model.fit(X_train, Y_train) # The model is trained on the entire training dataset.
train_result = mean_squared_error(res_ABR.predict(X_train), Y_train)
train_results.append(train_result)
test_result = mean_squared_error(res_ABR.predict(X_test), Y_test)
test_results.append(test_result)
print(name)
print('Mean Cross Validation Score: ', cv_results.mean())
print('Std Cross Validation Score: ', cv_results.std())
print('Train MSE: train_result: ', train_result)
print('Test MSE: test_result: ', test_result)

# We loop through all the models to compare the RMSE and the MSE
 
fig = plt.figure(figsize = [16, 8])
fig.suptitle("Algorithm Comparison: Results of K-Fold Cross Validation")
ax = fig.add_subplot(111)
plt.boxplot(kfold_results)  #from TRAINING result - THAT'S WHY THERE ARE 10
ax.set_xticklabels(names)
plt.show()


# ARIMA

X_train_ARIMA = X_train
X_test_ARIMA = X_test
train_len = len(X_train_ARIMA)
test_len = len(X_test_ARIMA)
total_len = len(X)

X_train_ARIMA = X_train
X_test_ARIMA = X_test
train_len = len(X_train_ARIMA)
test_len = len(X_test_ARIMA)
total_len = len(X)

modelARIMA =\
(stats.ARIMA(endog = Y_train,
                exog = X_train_ARIMA,
                order = [1, 0, 0]))
model_fit = modelARIMA.fit()

error_training_ARIMA =\
(    mean_squared_error(Y_train,
                       model_fit.fittedvalues)
)

predicted =\
(
    model_fit
    .predict(start = train_len - 1,
             end = total_len - 1,
             exog = X_test_ARIMA)[1: ]
)


error_testing_ARIMA =\
(    mean_squared_error(Y_test,
                        predicted)
)

train_results = [x for x in train_results if x is not None]
test_results = [x for x in test_results if x is not None]

test_results.append(error_testing_ARIMA)
train_results.append(error_training_ARIMA)
names.append("ARIMA")



# Algorithm Comparison

fig = plt.figure(figsize=[16, 8])
ind = np.arange(len(names))
width = 0.30
fig.suptitle("Comparing the Performance of Various Algorithms on the Training vs. Testing Data")
ax = fig.add_subplot(111)
plt.bar(ind - width/2, train_results, width=width, label="Errors in Training Set")
plt.bar(ind + width/2, test_results, width=width, label="Errors in Testing Set")
plt.legend()
ax.set_xticks(ind)
ax.set_xticklabels(names, rotation=45, ha='right')  # Set rotation to 45 degrees
plt.ylabel("Mean Squared Error (MSE)")
plt.show()


# Hyperparameter Tuning; Grid Search for ARIMA

def assess_ARIMA_model(arima_order):
    
    modelARIMA = stats.ARIMA(endog = Y_train, 
                             exog = X_train_ARIMA,
                             order = arima_order)
    # Our model takes an arima_order as input, 
    # fits an ARIMA model to the training data Y_train 
    # with exogenous variables X_train_ARIMA, 
    
    model_fit = modelARIMA.fit()
    # and then calculates 

    error = mean_squared_error(Y_train,
                               model_fit.fittedvalues)
    
    # and returns the Mean Squared Error (MSE) 
    # between the true and the fitted values.

    return error

def assess_models(p_values, d_values, q_values):
    
    # Team, our function performs grid search 
    # over all combinations of provided p, d, and q values. 
    
    # For each combination, it calculates the MSE and prints it. 
    
    # If the MSE for the current combination 
    # is less than the best score encountered so far, 
    # it updates the best score and the corresponding configuration. 
    
    # At the end of the grid search, 
    # it prints the best configuration and its MSE.
    
    best_score, best_cfg = float("inf"), None

    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p, d, q)
                try:
                    mse = assess_ARIMA_model(order)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                    
                    print("ARIMA%s MSE = %.7f" % (order, mse)
                          )
                    
                except:
                    continue
    print("Best ARIMA%s MSE = %.7f" % (best_cfg, best_score)
          )
    
# parameters to use for assessment

# Recall that the ARIMA model 
# is characterized by three parameters: 
# (p, d, q) which stand for the order of autoregression, 
# the degree of differencing, 
# and the order of the moving average, respectively.

p_values = [0, 1, 2]
d_values = range(0, 2)
q_values = range(0, 2)

assess_models(p_values, d_values, q_values)

ARIMA_Tuned =\
    stats \
    .ARIMA(endog = Y_train,
           exog = X_train_ARIMA,
          order = [0,0,1] # Input optimal set of hyperparameters here
          )

ARIMA_Fit_Tuned = ARIMA_Tuned.fit()

Predicted_Tuned =\
    model_fit \
    .predict(start = train_len - 1,
             end = total_len - 1,
             exog = X_test_ARIMA)[1:]
    
test_result_arima = print(mean_squared_error(Y_test, Predicted_Tuned))
train_result_arima = print(mean_squared_error(Y_train, model_fit.fittedvalues))
train_results.append(train_result_arima)
test_results.append(test_result_arima)

names = []
for name, model in models:
    names.append(name)

names.append('ARIMA-X')
 
train_results = [x for x in train_results if x is not None]
test_results = [x for x in test_results if x is not None]
result_list = [res_LN, res_LS, res_EN, res_DTR, res_KNR, res_SVR, res_RFR, res_ETR, res_GBR, res_ABR, Predicted_Tuned]
names.append('OLS')

model_OLS = sm.OLS(Y_train, X_train)
res_OLS = model_OLS.fit()
print(res_OLS.summary())

train_result = mean_squared_error(res_OLS.predict(X_train), Y_train)
train_results.append(train_result)
test_result = mean_squared_error(res_OLS.predict(X_test), Y_test)
test_results.append(test_result)

df_result = pd.DataFrame()

df_result['models'] = names
df_result['train_MSE'] = train_results
df_result['test_MSE'] = test_results
df_result.sort_values(by='test_MSE')


plt.figure(figsize = (16, 10)
           )

Predicted_Tuned.index = Y_test.index

plt.plot(np.exp(Y_test).cumprod() - 1, "black", label = "Actual Y")

# plt.plot(np.exp(Predicted_Tuned).cumprod(), "r--", label = "Predicted Y (Y hat) - ARIMA-X")
# plt.plot(Y_test.index, np.exp(res_EN.predict(X_test)).cumprod(), "orange", label = "Predicted Y (Y hat) - Elastic Net")
# plt.plot(Y_test.index, np.exp(res_LN.predict(X_test)).cumprod(), 'b--',  label = "Predicted Y (Y hat) - Linear Regression")
# plt.plot(Y_test.index, np.exp(res_LS.predict(X_test)).cumprod(), label = "Predicted Y (Y hat) - Lasso")
# plt.plot(Y_test.index, np.exp(res_DTR.predict(X_test)).cumprod(), label = "Predicted Y (Y hat) - Decision Tree Regression")
# plt.plot(Y_test.index, np.exp(res_KNR.predict(X_test)).cumprod(), label = "Predicted Y (Y hat) - KNeighborsRegressor")
# plt.plot(Y_test.index, np.exp(res_SVR.predict(X_test)).cumprod(), label = "Predicted Y (Y hat) - SVR")
# plt.plot(Y_test.index, np.exp(res_RFR.predict(X_test)).cumprod(), label = "Predicted Y (Y hat) - Random Forest Regressor")
# plt.plot(Y_test.index, np.exp(res_ETR.predict(X_test)).cumprod(), label = "Predicted Y (Y hat) - Extra Trees Regressor")
# plt.plot(Y_test.index, np.exp(res_GBR.predict(X_test)).cumprod(), label = "Predicted Y (Y hat) - Gradient Boosting Regressor")
# plt.plot(Y_test.index, np.exp(res_ABR.predict(X_test)).cumprod(), label = "Predicted Y (Y hat) - Ada Boost Regressor")

plt.plot(Y_test.index, np.exp(res_OLS.predict(X_test)).cumprod() -1, 'g--', label = "Predicted Y (Y hat) - OLS")

plt.legend()

plt.show()
