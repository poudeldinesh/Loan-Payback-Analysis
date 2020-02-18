# Loan Payback Analysis( Business Intelligence)
Loan delinquency and loan default are becoming a growing problem for the banking and other financial institutions. Failure to recover loans on time puts these loan lending institutions at financial loss. This eventually makes it difficult for people who genuinely need loan but never had any banking history. While it is very difficult to accurately claim who will pay back the loan, it is possible to estimate the probability of person paying back the loan.

In order to help answer this question, we obtained data posted by Home Credit on Kaggle. Home credit, a non-bank financial institution focuses primarily on lending loans to people with little or no credit history. 

The main objective of this project is to prepare a dataset that has necessary variables to predict how likely a person would repay a loan. Correlation studies was also performed between some of the variables in final dataset to check if there existed any significant relation between those variables as this may help with selection of variables for machine learning model. The results from this study can also be used to identify the factors that are common to people who face a difficulty in paying back the loan. 

#### Processes involved:
1. Data Selection
2. Data Processing & Visulaization( Tableau )
3. Data Merging( 4 different datasets from different sources)
4. Data CLeaning
5. Analysis

#### Datasets:
1. Loan Application Dataset
2. Credit Bureau Dataset
3. Credit Card Dataset
4. Previous Loan Application Dataset

#### Instruction to Run Code
1. Download four Datasets from https://drive.google.com/drive/folders/1ni6ABr423eddRWm9njLzkCe3Jffdt5rO?usp=sharing
2. Value of variable “path” should be changed to location where all 4 datasets are downloaded.
3. Import libraries: pandas, numpy, os, scipy.stats & re
4. Run the "Loan_Payback_Analysis_code" on Python IDE.

#### Output
Clean merged dataset with choosen variables that can be used with Machine Learning techniques to predict the probability of person paying back the loan.
