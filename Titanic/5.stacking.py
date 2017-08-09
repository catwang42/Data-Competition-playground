import pandas as pd 
import numpy as np 
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC 
import xgboost as xgb
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
import matplotlib.pyplot as plt
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

##########################################################
#1.reads the data 
filepath = '/Users/zhizhenwang/Desktop/kaggle/titanic/data/'
df_all = pd.read_csv(filepath+'test.csv')
topfeature_train = pd.read_csv(filepath+'topfeature_train.csv')
topfeature_test_X = pd.read_csv(filepath+'topfeature_test.csv')


##########################################################
#2. training and testing set 
topfeature_train_X = topfeature_train.drop(['Survived'], axis = 1)
topfeature_train_y = topfeature_train['Survived'].ravel()


##########################################################
#3. Set up model and params 
#Random Forest 0.824022346369
#==============================================================================
# rf_params ={
#     'max_depth':[20], 
#     'min_samples_split':[3], 
#     'min_samples_leaf':[2],     
#     'verbose':[0]
# }
#==============================================================================
#0.832960893855
rf_param_grid = { 'min_samples_split':[2], 'max_depth':[20]}
rf_clf = RandomForestClassifier(n_estimators = 500,random_state=42, n_jobs=50, criterion = 'gini', max_features='sqrt')

#Gradient Boosting 0.824022346369
gbm_params={
    'learning_rate': [0.1],
    'min_samples_split': [2],
    'min_samples_leaf': [2], 
    'max_depth': [6]
}
gbm_clf = GradientBoostingClassifier(n_estimators = 500, loss='exponential', max_features='sqrt',random_state=42,verbose=0)

#Extra Trees 0.82905027933
et_params={
    'max_depth': [8]

}
et_clf = ExtraTreesClassifier(n_estimators = 500, max_features = 'sqrt', max_depth = 8,
                              n_jobs = 50, criterion = 'entropy', random_state = 42, verbose = 0)

#Ada Boosting 0.835754189944 
ada_params = {
    'algorithm' : ['SAMME.R']
}
ada_clf = AdaBoostClassifier(random_state=42,n_estimators = 500,learning_rate=0.75 )

#SVC
svc_params = {
    'kernel' : ['linear'],
    'C' : [0.025],
    'gamma':[0.001, 0.01, 0.1]
    }
svc_clf=SVC()

##########################################################
#4.Stratified shuffle split + Gridsearch
stratifiedCV = StratifiedShuffleSplit(n_splits = 10, test_size=0.2, random_state =0)

def grid_cv(clf,param,name):
    grid_search = GridSearchCV(clf,param,verbose = 3, scoring ='accuracy', cv = stratifiedCV)
    grid_search.fit(topfeature_train_X,topfeature_train_y)
    print(name, "Best Params:" + str(grid_search.best_params_))
    print(name, "Best Score:" + str(grid_search.best_score_))
    return grid_search.predict(topfeature_test_X), grid_search.predict(topfeature_train_X)

    
rf_predict, rf_train_predict = grid_cv(rf_clf, rf_param_grid, 'randomForest')   #0.832960893855
gbm_predict, gbm_train_predict = grid_cv(gbm_clf, gbm_params, 'GradientBoosting')   #0.824022346369
ada_predict, ada_train_predict = grid_cv(ada_clf,ada_params,'AdaBoosting')  #0.835754189944
et_predict, et_train_predict = grid_cv(et_clf,et_params,'ExtraTree') #0.82905027933#
#svc_predict, svc_train_predict = grid_cv(svc_clf,svc_params,'SVC')


##########################################################
#5.First level prediction result table 
base_predictions_train = pd.DataFrame( {'RandomForest': rf_train_predict,
     'GradientBoosting': gbm_train_predict,
     'AdaBoosting': ada_train_predict,
     'ExtraTree': et_train_predict
     #'SVC':svc_train_predict
    })
base_predictions_train.head(5)

#heatmap of model result 
data = [
    go.Heatmap(
        z= base_predictions_train.astype(float).corr().values ,
        x=base_predictions_train.columns.values,
        y= base_predictions_train.columns.values,
          colorscale='Portland',
            showscale=True,
            reversescale = True
    )
]
py.iplot(data, filename='labelled-heatmap')

#new training and testing set for level 2 model xgboost
x_train = np.concatenate(( rf_train_predict.reshape(-1, 1), gbm_train_predict.reshape(-1, 1), 
                          ada_train_predict.reshape(-1, 1), et_train_predict.reshape(-1, 1)), axis=1)
x_test = np.concatenate(( rf_predict.reshape(-1, 1), gbm_predict.reshape(-1, 1), 
                         ada_predict.reshape(-1, 1), et_predict.reshape(-1, 1)), axis=1)

xgb_clf = xgb.XGBClassifier(n_estimators=2000,max_depth=4,min_child_weight=2,gamma=0.9,colsample_bytree=0.8,
                              objective='binary:logistic', nthread=-1,scale_pos_weight=1).fit(x_train,topfeature_train_y)
xgb_prediction = xgb_clf.predict(x_test)

############################################################
#5. submission 
PassengerId = df_all['PassengerId']
StackingSubmission = pd.DataFrame({ 'PassengerId': PassengerId,
                            'Survived': xgb_prediction })
StackingSubmission.to_csv(filepath+"StackingSubmission.csv", index=False)