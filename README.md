# Customer-Churn-Analysis-Prediction-
This is my first Capstone Project 

## OVERVIEW

The goal of this project is to build a machine learning model that can accurately predict whether the customer will leave the bank or not i.e. Exited or not. In this project, the focus is set on predicting whether a specific customer will continue to use the bank’s services or not. This allows the bank to determine the factors that lead to customers leaving their services for other financial services, and an in-depth analysis can help the financial institutions retain the customers.

The dataset provided had 10,000 rows and 14 attributes such as Credit Score, Gender, Age, Tenure, Estimated Salary, and more. A classifier is being built to determine which customers will Exit and which will not. Here, I have to deal with a **classification problem** whose main focus is to predict Discrete Values.

## PROBLEM STATEMENT

The project focuses on providing a data-driven solution to predict whether a customer will churn or not.

## DATA CLEANING

While working on the pre-defined datasets, I did data cleaning of my dataset. The dataset consisted of a few unnecessary columns such as Row Number, Customer ID, and Surname which added little value in classifying the customers. These were dropped, as these only added noise to the dataset. It was also checked whether the dataset had any missing values, which it did not have.
The next task was to categorize the numerical and categorical variables. The categorical variables were Geography and Gender, which were both One-Hot-encoded. This data was then passed to the classifier, with the ground truth being the “Exited” attribute and all the remaining columns being the inputs.

## POWER BI DASHBOARD

![Bank customer Churn Dashboard](https://github.com/muskansinghal14/Customer-Churn-Analysis-Prediction-/assets/140623673/24d1efcd-5d17-4f0d-b487-ef6174fcc331)

## Findings from Power BI Dashboard

1) The churn rate is very high at 56.2% for middle-aged customers between 51-60 years old.
2) In terms of those with credit card facilities, the churn rate is highest amongst the lowest credit score group of >400.
3) Surprisingly, the customers whose account balances are >=200k are the most churned in terms of account balance. 
4) The overall churn rate for the male population is 16.5 whereas the overall churn rate for the female population is 25.1%. 
5) For the products, the churn rate is high @ 27.7% amongst customers in the Prod 1 group. The age account and credit score factors are the same here. Similar situation for prod 2 =, however in prod 2, the churn rate is at all times 100% for customers between the 1k-1ok account balance. 
6) For customers who own the credit card, for them the churn rate is highest for the 51-60 age groups. The credit score and account balance factors are the same here. 
7) Inactive customers have the highest chance of churning so in the report I created I can see that customers with age groups of 51-60 and 61-70 have the highest churn rate, credit scores less than 400 have the highest churn rate for inactive customers and account balance from 1k-10k has highest churn rate. 
8) In Summary, age group of 51–60, people with low credit scores and those with high account balances are most likely to churn. The bank should consider effective strategies that can address the findings, to better customer retention and reduce churn rate. 

## CLASSIFICATION MODEL

1) As the data was dependent on multiple factors or features, so for that, I used 4 algorithms to make a classification report, confusion matrix, and accuracy score to predict my model and to see which classification algorithms among these gave me the best accuracy.  
2) The test accuracy of the Decision Tree is 90.6%, the Gradient Boosting Classifier is 79.8%, and the Logistic Regression is 64.6% while the Random Forest Classifier had a test accuracy of 93.6%. Test accuracy here refers to the number of correctly predicted labels divided by the total number of labels.
3) After reviewing the results of the Logistic Regression, Decision Trees, and Gradient Boosting Classifier, it was evident that the dataset could perform better on a classifier that deals with complex attributes and multiple dependencies. This is why the Random Forest Classifier was chosen, and the results showed an improvement. 
4) Since the data imbalance existed, The F1 score was also calculated to see how well the classifier performed on individual classes. The Decision Tree had an F1-score of 0.90 and 0.91 while the Random Forest Classifier had an F1-score of 0.93 and 0.94. 

## CONCLUSION

Developed a classifier i.e. Random Forest Classifier that allowed me to predict whether a customer will leave or not with an accuracy of  93.6%. This can help in identifying potentially vulnerable customers who are planning to leave, and we can act in ways to retain them. 












































