import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

#Loading Dataset
df = pd.read_excel('Sample_Superstore.csv')

print(df.isnull().sum())

print(df.info())