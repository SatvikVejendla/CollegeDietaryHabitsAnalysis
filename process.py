import pandas as pd
import numpy as np

df = pd.read_csv("food_coded.csv")

#Feature Engineered Variable Extracted
df = df[["GPA", "calories_day", "comfort_food_reasons_coded", "cook", "cuisine", "diet_current_coded", "eating_changes_coded", "eating_changes_coded1", "eating_out", "employment", "exercise", "fruit_day", "healthy_feeling", "income", "marital_status", "nutritional_check", "self_perception_weight", "sports", "vitamins", "weight"]]

print(df.head())
print(df.shape)

print(df.dtypes)

#We see from this result that we need to process the GPA and Weight columns to turn them into numeric columns


#Cleaning GPA column
print(df.GPA.value_counts())

#Replacing string unknown rows with NaN
df[df["GPA"] == "Personal "] = np.nan
df[df["GPA"] == "Unknown"] = np.nan

#Rounding all GPAs to 1 decimal place so we get better representation of the data
df["GPA"] = df["GPA"].astype(float).round(1)
print(df.GPA.value_counts())


#Cleaning Weight Columns

print(df.weight.value_counts())

df[df["weight"] == "nan "] = np.NaN
df["weight"] = df["weight"].astype(float)

#Rounding all weights to the nearest 5lbs so we get better representation of the data
df["weight"] = round(df["weight"]/5.0)*5.0

print(df.weight.value_counts())


x = df.columns.to_series().groupby(df.dtypes).groups
for i in x:
    print()
    print(str(i) + ": " +  ', '.join(x[i].to_numpy()))

#We see that all data has been processed to numeric formats. Now, we need to account for the NaN values.

#Find how many rows have NaN values
def findNaN(df):
    numNull = 0
    for index, row in df.iterrows():
        #iterate entries of row
        for entry in row:
            if pd.isnull(entry):
                numNull += 1
                break
    
    print(str(numNull) + "/" + str(df.shape[0]) + " rows have NaN values")


findNaN(df)

#This number is too high for us to just drop the rows with NaN values. We will need to impute the data.

#We will use the KNNImputer to impute the data


from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5)

df = pd.DataFrame(imputer.fit_transform(df), columns = df.columns)

print(df.head())

findNaN(df)

#Data has been imputed but the problem is that certain categorical columns are now continuous because of the imputed data. We need to round them back.

df["calories_day"] = df["calories_day"].round(0)
df["comfort_food_reasons_coded"] = df["comfort_food_reasons_coded"].round(0)
df["cook"] = df["cook"].round(0)
df["cuisine"] = df["cuisine"].round(0)
df["diet_current_coded"] = df["diet_current_coded"].round(0)
df["eating_changes_coded"] = df["eating_changes_coded"].round(0)
df["eating_changes_coded1"] = df["eating_changes_coded1"].round(0)
df["eating_out"] = df["eating_out"].round(0)
df["employment"] = df["employment"].round(0)
df["exercise"] = df["exercise"].round(0)
df["fruit_day"] = df["fruit_day"].round(0)
df["healthy_feeling"] = df["healthy_feeling"].round(0)
df["income"] = df["income"].round(0)
df["marital_status"] = df["marital_status"].round(0)
df["nutritional_check"] = df["nutritional_check"].round(0)
df["self_perception_weight"] = df["self_perception_weight"].round(0)
df["sports"] = df["sports"].round(0)
df["vitamins"] = df["vitamins"].round(0)

df["GPA"] = df["GPA"].round(1)
df["weight"] = round(df["weight"]/5.0)*5.0

print(df.head())
print(df.shape)

df.to_csv("processed_data.csv", index=False)