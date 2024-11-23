import pandas as pd
import numpy as np
import statsmodels.api as sm #import the statsmodels library to use the linear regression model
#from linearmodels.iv import IV2SLS
#from linearmodels.panel import PanelOLS
import matplotlib.pyplot as plt
import seaborn as sns


np.random.seed(42)
n = 500 #Number of the observations in the dataset

#instrumental variable:  internet_usuage rate is used a sthe instrumental variable for the exogenous variable of the sharenting
internet_usuage = np.random.normal(50, 100, n) # internet usuage rate is normally distributed with the mean of 50 and the standard deviation of 100 with the 500 nnumber of the observations and the inernet usage is the instrument variable for the sharenting

#endogenous variable : sharenting is the endogenous variable in the dataset
sharenting = 0.5 * internet_usuage + np.random.normal (0, 10, n) #sharenting  is the endogenous variable that is correlated the error term in our case and np.random is the error term normallly distributed with 0 mean here

#control independent variables in ours dataset to find out the the effect overall on the child behaviour while growing up
parents_income = np.random.normal(50000, 10000, n) #parents_income is the exogenous variable in the dataset and normally distributed with 50000 mean and 10000 standard deviation.

#control independent variables to check how much of the parents education is effecting the child behaviour while growing up
parents_education = np.random.normal(12, 3, n)

#dependent variable : we are here trying to find out the effect of the sharenting and the
child_outcome = 0.4 * sharenting + 0.3 *parents_education + 0.2*parents_education + np.random.normal(0,1 , n) #child_outcome is the dependent variable in the dataset and normally distributed with 0 mean and 1 standard deviation

#Generate the synthetic data for the analysis
data = pd.DataFrame ({
    'sharenting' : sharenting, #sharenting is the endogenous variable in the dataset
    'internet_usuage' : internet_usuage ,#internet_usuage is the instrumental variable in the dataset
    'parents_income' : parents_income, #parents_income is the exogenous variable in the dataset
    'parents_education': parents_education, #parents_education is the exogenous variable in the dataset
    'child_outcome' : child_outcome ,#child_outcome is the dependent variable in the dataset
})

#Display the first 5 rows of the dataset
print(data.head())

#2sls model to estimate the effect of the sharenting on the child outcome
#First stage regression
data['constant'] = 1
First_stage = sm.OLS(data['sharenting'], data['constant', 'internet_ususage'])
First_stage_result = First_stage.fit()
print(First_stage_result.summary())

#predicted sharenting
data['predicted_sharenting']= First_stage_result.fittedvalues #predicted sharenting

Second_stage = sm.OLS(data['child_outcome'], data['constant', 'predicted_sharenting', 'parents_income', 'parents_education'])
Second_stage_result =Second_stage.fit()
print(Second_stage_result.summary())

#IV2SLS method to estimate the effect of the sharenting on the child outcome
iv_model = IV2SLS(
    data['child_outcome'],
    endog = data['sharenting'],
    exog = data['parents_income', 'parents_education', 'constant'],
    instrument = data['internet_usuage']
).fit()

print(iv_model.summary)



#visulalize the relationship between the sharenting and the child_outcome. Interesting enough you are here trying to figure out the relationship it has between the sharenting and the child income in this case
sns.scatterplot(x=data['sharenting'], y=data['child_outcome'])
plt.title('Relationship between the sharenting the child outcome')
plt.xlabel('sharenting')
plt.ylabel('Child Outcome')
plt.show()

