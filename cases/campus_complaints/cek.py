import pandas as pd

df_raw = pd.read_csv('cases/campus_complaints/indonesian_campus_complaints_44k.csv')
df_train = pd.read_csv('cases/campus_complaints/train.csv')
df_val = pd.read_csv('cases/campus_complaints/val.csv')

print(df_raw.shape)
print(df_raw.head())

print(df_train.shape)
print(df_train.head())

print(df_val.shape)
print(df_val.head())

print(df_raw.duplicated().sum())