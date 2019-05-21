# Importing Library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import warnings
from sklearn.preprocessing import LabelEncoder
import csv

# Reading the training dataset in a dataframe using Pandas
df = pd.read_csv("train.csv")
"""
# First 10 Rows of training Dataset
print("printing top 10 row of data set")
print(df.head(10))

#taking input from user
print("enter details of the applicant")
id11="LP00"
inp=input("Enter 4 digit loan id ")
id1=id11+inp
gender=input("Enter gender (Male/Female)")
married=input("Enter marital status (Yes/No)")
dependent=input("enter no of dependent (1/2/3+)")
education=input("enter education qualification (Graduate/Not Graduate)")
employment=input("enter whether Self_Employed (Yes/No)")
income=int(input("enter applicant income "))
income2=int(input("enter co-applicant income if not there then enter zero "))
amount=int(input("enter loan amount you want "))
term=int(input("enter loan amount term in days "))
hist=int(input("enter credit history(1 if you have credit history else 0)" ))
area=input("enter property area(Urban/Semiurban/Rural) ")
mylist=[id1,gender,married,dependent,education,employment,income,income2,amount,term,hist,area]
with open('test.csv','a') as fd:
    writer = csv.writer(fd)
    writer.writerow(mylist)
"""
# Reading the test dataset in a dataframe using Pandas
test = pd.read_csv("test.csv")

# Store total number of observation in training dataset
df_length =len(df)
print("total no of observation in training dataset is ",df_length)

# Store total number of columns in testing data set
test_col = len(test.columns)
print("no of columns in training data set is ",test_col )
# # Understanding the various features (columns) of the dataset.
print("Description about data set")
print(df.describe())

# Get the unique values and their frequency of variable Property_Area
print("unique values and their frequency of variable Property_Area ")
print(df['Property_Area'].value_counts())
           
# Histogram of variable ApplicantIncome
print("Plotting Histogram for applicant income")
plt.hist(df['ApplicantIncome'])
plt.ylabel("No of people")
plt.show()

# Histogram of variable LoanAmount
print("Plotting Histogram for Loan amount")
loanmean=df['LoanAmount'].mean()
df['LoanAmount']=df['LoanAmount'].fillna(loanmean)
plt.hist(df['LoanAmount'])
plt.ylabel("No of people")
plt.show()

# Loan approval rates in absolute numbers
loan_approval = df['Loan_Status'].value_counts()['Y']
print("total no loan approved application",loan_approval)
# Credit History and Loan Status
pd.crosstab(df ['Credit_History'], df ['Loan_Status'], margins=True)
#Function to output percentage row wise in a cross table
def percentageConvert(ser):
    return ser/float(ser[-1])
# Loan approval rate for customers having Credit_History (1)
df1=pd.crosstab(df ["Credit_History"], df ["Loan_Status"], margins=True).apply(percentageConvert, axis=1)
loan_approval_with_Credit_1 = df1['Y'][1]

print("Loan approval rate for customers having Credit_History 1 is",loan_approval_with_Credit_1*100)

#credit_history for loan approved
print("\n\ncredit history for loan approved")
print(df1['Y'])

# Replace missing value of Self_Employed with more frequent category
df['Self_Employed']=df['Self_Employed'].fillna('No',inplace=True)
# Add both ApplicantIncome and CoapplicantIncome to TotalIncome
df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']

# Looking at the distribtion of TotalIncome
df['LoanAmount']=df['LoanAmount'].fillna(df['LoanAmount'].mean())
# Perform log transformation of LoanAmount to make it closer to normal
df['LoanAmount_log'] = np.log(df['LoanAmount'])

# Impute missing values for Gender
df['Gender'].fillna(df['Gender'].mode()[0],inplace=True)
# Impute missing values for Married
df['Married'].fillna(df['Married'].mode()[0],inplace=True)
# Impute missing values for Dependents
df['Dependents'].fillna(df['Dependents'].mode()[0],inplace=True)
# Impute missing values for Credit_History
df['Credit_History'].fillna(df['Credit_History'].mode()[0],inplace=True)

# Convert all non-numeric values to number
cat=['Gender','Married','Dependents','Education','Self_Employed','Credit_History','Property_Area']

for var in cat:
    le = preprocessing.LabelEncoder()
    df[var]=le.fit_transform(df[var].astype('str'))
print("data types after Label encoding")
print(df.dtypes)

#Import models from scikit learn module:
from sklearn import metrics
from sklearn.cross_validation import KFold

#classification_model(model, df,predictors_Logistic,outcome_var)
def classification_model(model, data, predictors, outcome):
    #Fit the model:
    model.fit(data[predictors],data[outcome])
  
    #Make predictions on training set:
    predictions = model.predict(data[predictors])
  
    #Print accuracy
    accuracy = metrics.accuracy_score(predictions,data[outcome])
    print ("Accuracy : %s" % "{0:.3%}".format(accuracy))

    #Perform k-fold cross-validation with 5 folds
    kf = KFold(data.shape[0], n_folds=5)
    error = []
    for train, test in kf:
        # Filter training data
        train_predictors = (data[predictors].iloc[train,:])
    
        # The target we're using to train the algorithm.
        train_target = data[outcome].iloc[train]
    
        # Training the algorithm using the predictors and target.
        model.fit(train_predictors, train_target)
    
        #Record error from each cross-validation run
        error.append(model.score(data[predictors].iloc[test,:], data[outcome].iloc[test]))
 
    print ("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error)))

    #Fit the model again so that it can be refered outside the function:
    model.fit(data[predictors],data[outcome])

#Create a flag for Train and Test Data set
df['Type']='Train' 
test['Type']='Test'
fullData = pd.concat([df,test],axis=0, sort=True)

#Look at the available missing values in the dataset
print("available missing values are")
print(fullData.isnull().sum())

#Identify categorical and continuous variables
ID_col = ['Loan_ID']
target_col = ["Loan_Status"]
cat_cols = ['Credit_History','Dependents','Gender','Married','Education','Property_Area','Self_Employed']

#Imputing Missing values with mean for continuous variable
fullData['LoanAmount'].fillna(fullData['LoanAmount'].mean(), inplace=True)
fullData['LoanAmount_log'].fillna(fullData['LoanAmount_log'].mean(), inplace=True)
fullData['Loan_Amount_Term'].fillna(fullData['Loan_Amount_Term'].mean(), inplace=True)
fullData['ApplicantIncome'].fillna(fullData['ApplicantIncome'].mean(), inplace=True)
fullData['CoapplicantIncome'].fillna(fullData['CoapplicantIncome'].mean(), inplace=True)

#Imputing Missing values with mode for categorical variables
fullData['Gender'].fillna(fullData['Gender'].mode()[0], inplace=True)
fullData['Married'].fillna(fullData['Married'].mode()[0], inplace=True)
fullData['Dependents'].fillna(fullData['Dependents'].mode()[0], inplace=True)
fullData['Loan_Amount_Term'].fillna(fullData['Loan_Amount_Term'].mode()[0], inplace=True)
fullData['Credit_History'].fillna(fullData['Credit_History'].mode()[0], inplace=True)

#Create a new column as Total Income

fullData['TotalIncome']=fullData['ApplicantIncome'] + fullData['CoapplicantIncome']
fullData['TotalIncome_log'] = np.log(fullData['TotalIncome'])

#Histogram for Total Income
print("Plotting Histogram for fulldata Total income")
plt.hist(fullData['TotalIncome'])
plt.ylabel("No of people")
plt.show()

#create label encoders for categorical features
for var in cat_cols:
    number = LabelEncoder()
    fullData[var] = number.fit_transform(fullData[var].astype('str'))

train_modified=fullData[fullData['Type']=='Train']
test_modified=fullData[fullData['Type']=='Test']
train_modified["Loan_Status"] = number.fit_transform(train_modified["Loan_Status"].astype('str'))
warnings.filterwarnings('ignore')

#let’s make our model with 'Credit_History','Education','TotalIncome' & 'Dependents'
from sklearn.linear_model import LogisticRegression
predictors_Logistic=['Credit_History','Education','TotalIncome','Dependents']
x_train = train_modified[list(predictors_Logistic)].values
y_train = train_modified["Loan_Status"].values
x_test=test_modified[list(predictors_Logistic)].values
# Create logistic regression object
model = LogisticRegression()
# Train the model using the training sets
model.fit(x_train, y_train)
import seaborn as sns
print("plotting logistic regression for credit history vs loan status")
credit_hist=['Credit_History']
sns.regplot(x=train_modified[list(credit_hist)].values, y=y_train, data=df, logistic=True)
plt.xlabel("Credit history")
plt.ylabel("loan status")
plt.show()
#Predict Output
predicted= model.predict(x_test)
#Reverse encoding for predicted outcome
predicted = number.inverse_transform(predicted)
#Store it to test dataset
test_modified['Loan_Status']=predicted
outcome_var = 'Loan_Status'
classification_model(model, df,predictors_Logistic,outcome_var)
test_modified.to_csv("Logistic_Prediction.csv",columns=['Loan_ID','Loan_Status'])

with open('Logistic_Prediction.csv', 'r') as f:
    for row in reversed(list(csv.reader(f))):
        id2=row[1]
        res=row[2]
        break;
no="N"
print("for LOAN_ID",id2)
if(res==no):
    print("loan can not be sanctioned")
else:
    print("loan can be sanctioned")
 