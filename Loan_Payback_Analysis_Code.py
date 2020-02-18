
"""
Created on Sat Sep 28 16:39:06 2019

@ISQS6339-001-2019-GROUP-3
Author: Dinesh Poudel
        Aman Panwar
        Anurag Sharma
"""

# Import libraries
import pandas as pd
import numpy as np
import os
import scipy.stats as ss
import re

#define the input and output local computer file path
os.chdir("C:\\Users\\Dinesh Poudel\\Desktop\\Risk Analysis")


# import all four datasets from local computer
df_application=pd.read_csv("application_train.csv")
df_bureau=pd.read_csv("bureau.csv")
df_credit_balance=pd.read_csv("credit_card_balance.csv")
df_previous=pd.read_csv("previous_application.csv")

# Select only needed columns from loan applicant's data
df= df_application[['SK_ID_CURR', 'TARGET', 'NAME_CONTRACT_TYPE','CODE_GENDER',
                    'FLAG_OWN_CAR', 'FLAG_OWN_REALTY','AMT_INCOME_TOTAL',
                    'AMT_CREDIT','AMT_GOODS_PRICE','NAME_INCOME_TYPE',
                    'NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS',
                    'REGION_POPULATION_RELATIVE','DAYS_BIRTH','DAYS_EMPLOYED',
                    'FLAG_CONT_MOBILE','OCCUPATION_TYPE','CNT_FAM_MEMBERS',
                    'REGION_RATING_CLIENT','ORGANIZATION_TYPE',
                    'AMT_REQ_CREDIT_BUREAU_YEAR']]

# Select only needed columns from bureau data
df1= df_bureau[['SK_ID_CURR','CREDIT_ACTIVE']]
df1= pd.get_dummies(df1, columns=['CREDIT_ACTIVE'])

#create new column for each unique category in "Credit Active" column
df1= df1.rename(columns={x: x.split('_')[0]+'_1_'+
    x.split('_')[2] for x in df1.columns[1:]})
# fill with aggregate number for each new column
df1= df1.groupby('SK_ID_CURR',as_index = False).sum()



# Select only needed columns credit balance data
df2= df_credit_balance[['SK_ID_CURR','AMT_BALANCE','AMT_CREDIT_LIMIT_ACTUAL']]
df2=df2.groupby(['SK_ID_CURR'],as_index = False)['AMT_BALANCE','AMT_CREDIT_LIMIT_ACTUAL'].sum()


# Select only needed columns previous application
df3= df_previous[['SK_ID_CURR','NAME_CONTRACT_STATUS']]
df3= pd.get_dummies(df3, columns=['NAME_CONTRACT_STATUS'])
df3= df3.rename(columns={x: x.split('_')[3]+'_1_'+x.split('_')[1]+'_'+x.split('_')[2] for x in df3.columns[1:]})
df3= df3.groupby('SK_ID_CURR',as_index = False).sum()

# left merge bureau data on loan application data
dft= df.merge(df1,on='SK_ID_CURR', how='left', indicator=True)

# for missing bureau data( person with no previous credit history), assign -1
dft.loc[dft['_merge']=='left_only',dft.columns[dft.columns.str.contains('CREDIT_1_')]] =-1 


# left merge credit balance data on mergered dataframe   
dft= dft.merge(df2,on='SK_ID_CURR', how='left', indicator='_merge2')

# for missing credit balance data( person with no previous credit history), assign -2
dft.loc[dft['_merge2']=='left_only',['AMT_BALANCE','AMT_CREDIT_LIMIT_ACTUAL']] =-2 

# left merged previous application data on mergered dataframe 
dft= dft.merge(df3,on='SK_ID_CURR', how='left', indicator='_merge3')

# for missing previous application data( person with no previous credit history), assign -3
dft.loc[dft['_merge3']=='left_only',dft.columns[dft.columns.str.contains('CONTRACT_STATUS')]] =-3


### check the columns with nan values.
nan_cols = [i for i in dft.columns if dft[i].isnull().any()]
dft.isnull().sum()

# filling missing values on categorical data
dft['OCCUPATION_TYPE'] = dft['OCCUPATION_TYPE'].fillna('Unknown')

# filling missing values with mean on numerical data
mean_value=dft['AMT_GOODS_PRICE'].mean()
dft['AMT_GOODS_PRICE']=dft['AMT_GOODS_PRICE'].fillna(mean_value)

# filling missing values with mean on numerical data
mean_value=dft['CNT_FAM_MEMBERS'].mean()
dft['CNT_FAM_MEMBERS']=dft['CNT_FAM_MEMBERS'].fillna(mean_value)

# filling missing values with number=0
dft['AMT_REQ_CREDIT_BUREAU_YEAR']=dft['AMT_REQ_CREDIT_BUREAU_YEAR'].fillna(0)


# file output
dft.to_csv("our_output_file.csv",index=False)

#filter out only inner join elements
dft_70 = dft[(dft != 'left_only').all(axis=1)]
#################################################################################
# correlation
#creates a list of all column names

all_col_dft=list(dft.columns.values) 

#creates a list of all continous variables
num_col= ['AMT_INCOME_TOTAL','AMT_CREDIT','AMT_GOODS_PRICE','REGION_POPULATION_RELATIVE',

'DAYS_BIRTH','DAYS_EMPLOYED',

'AMT_BALANCE','AMT_CREDIT_LIMIT_ACTUAL']

#converts target column to numpy array for Vcramer calculation
target_arr=dft['TARGET'].to_numpy()

#declares list of r for continous variables
list_cor_num=[]

#calculates r(correlation) b/w continous variable and target
for num_col in num_col:

    a=dft[num_col].to_numpy()

    s=str(ss.pointbiserialr(a, target_arr))

    s1=re.search(r"(?<==).*?(?=,)", s).group(0)

    list_cor_num.append(s1)

    print(num_col +':'+ s1)

##creates a list of all categorical variables
dft_cat_col = list(set(all_col_dft) - set(['SK_ID_CURR','TARGET',

'AMT_INCOME_TOTAL','AMT_CREDIT','AMT_GOODS_PRICE','REGION_POPULATION_RELATIVE',

'DAYS_BIRTH','DAYS_EMPLOYED',

'AMT_BALANCE','AMT_CREDIT_LIMIT_ACTUAL']))

#declares list of r for categorical variables
list_cor_cat=[]

#calculates r(correlation) b/w categorical variable and target
for dft_cat_col in dft_cat_col:

    confusion_matrix = pd.crosstab(dft['TARGET'], dft[dft_cat_col])
    confusion_matrix
    chi2 = ss.chi2_contingency(confusion_matrix)[0]

    n = confusion_matrix.sum().sum()

    phi2 = chi2/n

    r,k = confusion_matrix.shape

    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    

    rcorr = r - ((r-1)**2)/(n-1)

    kcorr = k - ((k-1)**2)/(n-1)

    l=np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))

    list_cor_cat.append(l)

    print(dft_cat_col +':'+ str(l))


##############################################################
# for df_70 only the inner join elements
##############################################################

#creates a list of all column names
all_col_dft_70=list(dft_70.columns.values) 


#creates a list of all continous variables
num_col= ['AMT_INCOME_TOTAL','AMT_CREDIT','AMT_GOODS_PRICE','REGION_POPULATION_RELATIVE',

'DAYS_BIRTH','DAYS_EMPLOYED',

'AMT_BALANCE','AMT_CREDIT_LIMIT_ACTUAL']

#converts target column to numpy array for Vcramer calculation
target_arr=dft_70['TARGET'].to_numpy()

#declares list of r for continous variables

list_cor_num=[]

#calculates r(correlation) b/w continous variable and target
for num_col in num_col:

    a=dft_70[num_col].to_numpy()

    s=str(ss.pointbiserialr(a, target_arr))

    s1=re.search(r"(?<==).*?(?=,)", s).group(0)

    list_cor_num.append(s1)

    print(num_col +':'+ s1)

##creates a list of all categorical variables

dft_cat_col = list(set(all_col_dft_70) - set(['SK_ID_CURR','TARGET',

'AMT_INCOME_TOTAL','AMT_CREDIT','AMT_GOODS_PRICE','REGION_POPULATION_RELATIVE',

'DAYS_BIRTH','DAYS_EMPLOYED',

'AMT_BALANCE','AMT_CREDIT_LIMIT_ACTUAL']))

    
#declares list of r for categorical variables

list_cor_cat=[]

#calculates r(correlation) b/w categorical variable and target

for dft_cat_col in dft_cat_col:

    confusion_matrix = pd.crosstab(dft_70['TARGET'], dft_70[dft_cat_col])
    confusion_matrix
    chi2 = ss.chi2_contingency(confusion_matrix)[0]

    n = confusion_matrix.sum().sum()

    phi2 = chi2/n

    r,k = confusion_matrix.shape

    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    

    rcorr = r - ((r-1)**2)/(n-1)

    kcorr = k - ((k-1)**2)/(n-1)

    l=np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))

    list_cor_cat.append(l)

    print(dft_cat_col +':'+ str(l))

##############################
# correlation matrix for left join dataframe    
f=dft.corr()

# correlation matrix for inner join dataframe    
g=dft_70.corr()

##########################
##END
##########################








