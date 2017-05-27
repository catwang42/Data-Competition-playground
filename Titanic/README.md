This is a very classic problem in Kaggle for newcomers to have a taste of what a Kaggle competition would be like. 

The problem we are trying to solve for this competition was to predict the survival possibility of a range of people, based on what we learn from the given features including Age, Pclass, Sex,  Fare, Carbin, etic. 
We can build our model based on training set (25000 labeled data), and then use our model to predict the binary result on the testing set, and our accuracy was measured by the correct hidden label in the test dataset. 


The first thing I do is to do some EDA (exploratory data analysis) on the training set and try to visualize the relationship between given features. 

I use python, and the library including Panda, Numpy, scikit learn, seabon, matplotlib to do the project. 
The initial insights I found including:
- some missing value for certain features (age, cabin)
- the average age of 2,3 class passenger was around 20s, and this number for the 1st class is 40+
- The 1st class passenger has higher survival rate compared to 2/3 class.
- the woman has a higher survival rate 
- cabin S has the highest survival rate , much higher than the rest of the cabin
etc

Some feature engineering around first insights:
- use RandomForest fit the missing age value (use given age as feature matrix and fit if to RandomForestRegressor, then use the 
- RandomForestRegressor to predict the age of people who has "null" value)
- use one-hot-encode to factorization cabin value. 
- use preprocessing model from scikit learn to scalling Age and Fare and ensure their value ranging from [-1,1]
- Last I choose simple yet classic Logistic Regression to serve as my model for this problem. 
- Convert feature into numpy array and fit it to the Model. 


The initial approach achieves only 0.75443 

Some of the features we are ignored in this submission and more feature engineering were needed to achieve higher score. 

Check the logistic regression coefficient, then we can have an ideal of which feature have higher weight, and i will do more work on that feature.  


some revised solution for feature engineering：
- Fit the age data differently, I classified them into 'Mr' 'Mrs', 'Miss' categories then using RandomForest to fit then (age and set are correlated )
- The age is not continue feature, change it to discrete feature
- use Pclass sand Sex as combined feature, they are largely related. 
- try some sentiment analysis in name feature,
- add the number of family member as the new feature. 

After the change, submisison accurancy incresed to -.80122