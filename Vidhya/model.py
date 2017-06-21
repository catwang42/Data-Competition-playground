## predictive model for loan analysis

import pandas as pd 
import numpy as np
import math as m 
#model from scikit learn 
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold   #For K-fold cross validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics

df = pd.read_csv('/Users/zhizhenwang/GoogleDrive/Project/Vidhya/cleandata.csv')

#Generic funciton for making a classificaiton mode and accessing performance 
def classification_model(model, data, predictors, outcome):
    #Fir model
    model.fit(data[predictors],data[outcome])
    
    #prediction 
    predictions = model.predict(data[predictors])
    
    #accuracy 
    accuracy = metrics.accuracy_score(predictions,data[outcome])
    print("Accuracy : %s" % "{0:.3%}".format(accuracy))
    
    #k-fold cross_validation with 5 folds
    kf = kFold(data.shape[0],n_folds=5)
    
    error=[]
    for train, test in kf:
        train_predictors = (data[predictors].iloc[train,:]) 
        train_target = data[outcome].iloc[train]
        model.fit(train_predictors,train_target)
        error.append(model.score(data[predictors].iloc[test,:],data[outcome].iloc[test]))
    
    print("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error)))
    
    model.fit(data[predictors],data[outcome])


#Applicants having a credit history (remember we observed this in exploration?)
#Applicants with higher applicant and co-applicant incomes
#Applicants with higher education level
#Properties in urban areas with high growth perspectives

##1.Logistic Regression 
outcome_var = 'Loan_Status'
model = LogisticRegression()
predictor_var = 'Credit_History'
classification_model(model,df,predictor_var,outcome_var)
#We can try different combination of variables:
predictor_var = ['Credit_History','Education','Married','Self_Employed','Property_Area']
classification_model(model, df,predictor_var,outcome_var)
## no increase after adding more feature 
#new feature / better model 


##2.Decision Tree 
model = DecisionTreeClassifier()
predictor_var = ['Credit_History','Gender','Married','Education']
classification_model(model, df,predictor_var,outcome_var)
#We can try different combination of variables:
predictor_var = ['Credit_History','Loan_Amount_Term','LoanAmount_log']
classification_model(model, df,predictor_var,outcome_var)
#overfitting 

##3.Random Forest
model = RandomForestClassifier(n_estimators=100)
predictor_var = ['Gender', 'Married', 'Dependents', 'Education',
       'Self_Employed', 'Loan_Amount_Term', 'Credit_History', 'Property_Area',
        'LoanAmount_log','TotalIncome_log']
classification_model(model, df,predictor_var,outcome_var)
#overfitting -> reduce predictor / turing parameter 
#Create a series with feature importances:
featimp = pd.Series(model.feature_importances_, index=predictor_var).sort_values(ascending=False)
print (featimp)
#choose top 5 
model = RandomForestClassifier(n_estimators=25, min_samples_split=25, max_depth=7, max_features=1)
predictor_var = ['TotalIncome_log','LoanAmount_log','Credit_History','Dependents','Property_Area']
classification_model(model, df,predictor_var,outcome_var)



