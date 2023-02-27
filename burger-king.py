import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd
data=pd.read_csv('burger-king-menu.csv')
print(data.columns)
print(data.describe())
print(data.info())
print(data.isna().sum())
print(data['Weight Watchers'])
print(data['Category'].value_counts())
column=data.columns.values
print(column)
for i in column[2:]:
    print('The Minimum value: ',data[i].min())
    print('The maximum value: ',data[i].max())
    print('The Skewness is: ',data[i].skew())
    print('The Kurtosis is: ',data[i].kurtosis())
    print('The columns description is: ','\n',data[i].describe())
sn.countplot(data.Category)
plt.show()
sn.pairplot(data[column[2:]])
plt.show()
break_fast=data[data.Category=='Breakfast']
median_cal=break_fast['Calories'].median()
less_df=break_fast[break_fast['Calories']< median_cal]
greater_df=break_fast[break_fast['Calories']>median_cal]
plt.plot(less_df.Calories,marker='o',color='pink',label=f'less than {median_cal}')
plt.plot(greater_df.Calories,marker='o',color='red',label=f'greater than {median_cal}')
plt.title('The median values of Calories in Break Fast')
plt.legend()
plt.show()

median_cal=break_fast['Fat Calories'].median()
less_df=break_fast[break_fast['Fat Calories']< median_cal]
greater_df=break_fast[break_fast['Fat Calories']>median_cal]
plt.plot(less_df['Fat Calories'],marker='o',color='orange',label=f'less than {median_cal}')
plt.plot(greater_df['Fat Calories'],marker='o',color='wheat',label=f'greater than {median_cal}')
plt.title('The median values of Fat Calories in Break Fast')
plt.legend()
plt.show()

median_cal=break_fast['Fat (g)'].median()
less_df=break_fast[break_fast['Fat (g)']< median_cal]
greater_df=break_fast[break_fast['Fat (g)']>median_cal]
plt.plot(less_df['Fat (g)'],marker='o',color='maroon',label=f'less than {median_cal}')
plt.plot(greater_df['Fat (g)'],marker='o',color='firebrick',label=f'greater than {median_cal}')
plt.title('The median values of Fat  in Break Fast')
plt.legend()
plt.show()

median_cal=break_fast['Saturated Fat (g)'].median()
less_df=break_fast[break_fast['Saturated Fat (g)']< median_cal]
greater_df=break_fast[break_fast['Saturated Fat (g)']>median_cal]
plt.plot(less_df['Saturated Fat (g)'],marker='o',color='coral',label=f'less than {median_cal}')
plt.plot(greater_df['Saturated Fat (g)'],marker='o',color='tomato',label=f'greater than {median_cal}')
plt.title('The median values of Saturated Fat (g) in Break Fast')
plt.legend()
plt.show()

median_cal=break_fast['Trans Fat (g)'].median()
less_df=break_fast[break_fast['Trans Fat (g)']< median_cal]
greater_df=break_fast[break_fast['Trans Fat (g)']>median_cal]
plt.plot(less_df['Trans Fat (g)'],marker='o',color='chocolate',label=f'less than {median_cal}')
plt.plot(greater_df['Trans Fat (g)'],marker='o',color='peru',label=f'greater than {median_cal}')
plt.title('The median values of Trans Fat (g) in Break Fast')
plt.legend()
plt.show()

median_cal=break_fast['Cholesterol (mg)'].median()
less_df=break_fast[break_fast['Cholesterol (mg)']< median_cal]
greater_df=break_fast[break_fast['Cholesterol (mg)']>median_cal]
plt.plot(less_df['Cholesterol (mg)'],marker='o',color='red',label=f'less than {median_cal}')
plt.plot(greater_df['Cholesterol (mg)'],marker='o',color='salmon',label=f'greater than {median_cal}')
plt.title('The median values of Cholesterol (mg) in Break Fast')
plt.legend()
plt.show()

median_cal=break_fast['Sodium (mg)'].median()
less_df=break_fast[break_fast['Sodium (mg)']< median_cal]
greater_df=break_fast[break_fast['Sodium (mg)']>median_cal]
plt.plot(less_df['Sodium (mg)'],marker='o',color='cyan',label=f'less than {median_cal}')
plt.plot(greater_df['Sodium (mg)'],marker='o',color='deepskyblue',label=f'greater than {median_cal}')
plt.title('The median values of Sodium (mg) in Break Fast')
plt.legend()
plt.show()

median_cal=break_fast['Total Carb (g)'].median()
less_df=break_fast[break_fast['Total Carb (g)']< median_cal]
greater_df=break_fast[break_fast['Total Carb (g)']>median_cal]
plt.plot(less_df['Total Carb (g)'],marker='o',color='gold',label=f'less than {median_cal}')
plt.plot(greater_df['Total Carb (g)'],marker='o',color='cornsilk',label=f'greater than {median_cal}')
plt.title('The median values of Total Carb (g) in Break Fast')
plt.legend()
plt.show()

median_cal=break_fast['Dietary Fiber (g)'].median()
less_df=break_fast[break_fast['Dietary Fiber (g)']< median_cal]
greater_df=break_fast[break_fast['Dietary Fiber (g)']>median_cal]
plt.plot(less_df['Dietary Fiber (g)'],marker='o',color='royalblue',label=f'less than {median_cal}')
plt.plot(greater_df['Dietary Fiber (g)'],marker='o',color='violet',label=f'greater than {median_cal}')
plt.title('The median values of Dietary Fiber (g) in Break Fast')
plt.legend()
plt.show()

median_cal=break_fast['Sugars (g)'].median()
less_df=break_fast[break_fast['Sugars (g)']< median_cal]
greater_df=break_fast[break_fast['Sugars (g)']>median_cal]
plt.plot(less_df['Sugars (g)'],marker='o',color='plum',label=f'less than {median_cal}')
plt.plot(greater_df['Sugars (g)'],marker='o',color='olive',label=f'greater than {median_cal}')
plt.title('The median values of Sugars (g) in Break Fast')
plt.legend()
plt.show()

median_cal=break_fast['Protein (g)'].median()
less_df=break_fast[break_fast['Protein (g)']< median_cal]
greater_df=break_fast[break_fast['Protein (g)']>median_cal]
plt.plot(less_df['Protein (g)'],marker='o',color='indigo',label=f'less than {median_cal}')
plt.plot(greater_df['Protein (g)'],marker='o',color='blueviolet',label=f'greater than {median_cal}')
plt.title('The median values of Protein (g) in Break Fast')
plt.legend()
plt.show()

median_cal=break_fast['Weight Watchers'].median()
less_df=break_fast[break_fast['Weight Watchers']< median_cal]
greater_df=break_fast[break_fast['Weight Watchers']>median_cal]
plt.plot(less_df['Weight Watchers'],marker='o',color='crimson',label=f'less than {median_cal}')
plt.plot(greater_df['Weight Watchers'],marker='o',color='hotpink',label=f'greater than {median_cal}')
plt.title('The median values of Weight Watchers in Break Fast')
plt.legend()
plt.show()

break_fast=data[data.Category=='Burgers']
median_cal=break_fast['Calories'].median()
less_df=break_fast[break_fast['Calories']< median_cal]
greater_df=break_fast[break_fast['Calories']>median_cal]
plt.plot(less_df.Calories,marker='o',color='pink',label=f'less than {median_cal}')
plt.plot(greater_df.Calories,marker='o',color='red',label=f'greater than {median_cal}')
plt.title('The median values of Calories in Burger')
plt.legend()
plt.show()

median_cal=break_fast['Fat Calories'].median()
less_df=break_fast[break_fast['Fat Calories']< median_cal]
greater_df=break_fast[break_fast['Fat Calories']>median_cal]
plt.plot(less_df['Fat Calories'],marker='o',color='orange',label=f'less than {median_cal}')
plt.plot(greater_df['Fat Calories'],marker='o',color='wheat',label=f'greater than {median_cal}')
plt.title('The median values of Fat Calories in Burger')
plt.legend()
plt.show()

median_cal=break_fast['Fat (g)'].median()
less_df=break_fast[break_fast['Fat (g)']< median_cal]
greater_df=break_fast[break_fast['Fat (g)']>median_cal]
plt.plot(less_df['Fat (g)'],marker='o',color='maroon',label=f'less than {median_cal}')
plt.plot(greater_df['Fat (g)'],marker='o',color='firebrick',label=f'greater than {median_cal}')
plt.title('The median values of Fat  in Burger')
plt.legend()
plt.show()

median_cal=break_fast['Saturated Fat (g)'].median()
less_df=break_fast[break_fast['Saturated Fat (g)']< median_cal]
greater_df=break_fast[break_fast['Saturated Fat (g)']>median_cal]
plt.plot(less_df['Saturated Fat (g)'],marker='o',color='coral',label=f'less than {median_cal}')
plt.plot(greater_df['Saturated Fat (g)'],marker='o',color='tomato',label=f'greater than {median_cal}')
plt.title('The median values of Saturated Fat (g) in Burger')
plt.legend()
plt.show()

median_cal=break_fast['Trans Fat (g)'].median()
less_df=break_fast[break_fast['Trans Fat (g)']< median_cal]
greater_df=break_fast[break_fast['Trans Fat (g)']>median_cal]
plt.plot(less_df['Trans Fat (g)'],marker='o',color='chocolate',label=f'less than {median_cal}')
plt.plot(greater_df['Trans Fat (g)'],marker='o',color='peru',label=f'greater than {median_cal}')
plt.title('The median values of Trans Fat (g) in Burger')
plt.legend()
plt.show()

median_cal=break_fast['Cholesterol (mg)'].median()
less_df=break_fast[break_fast['Cholesterol (mg)']< median_cal]
greater_df=break_fast[break_fast['Cholesterol (mg)']>median_cal]
plt.plot(less_df['Cholesterol (mg)'],marker='o',color='red',label=f'less than {median_cal}')
plt.plot(greater_df['Cholesterol (mg)'],marker='o',color='salmon',label=f'greater than {median_cal}')
plt.title('The median values of Cholesterol (mg) in Burger')
plt.legend()
plt.show()

median_cal=break_fast['Sodium (mg)'].median()
less_df=break_fast[break_fast['Sodium (mg)']< median_cal]
greater_df=break_fast[break_fast['Sodium (mg)']>median_cal]
plt.plot(less_df['Sodium (mg)'],marker='o',color='cyan',label=f'less than {median_cal}')
plt.plot(greater_df['Sodium (mg)'],marker='o',color='deepskyblue',label=f'greater than {median_cal}')
plt.title('The median values of Sodium (mg) in Burger')
plt.legend()
plt.show()

median_cal=break_fast['Total Carb (g)'].median()
less_df=break_fast[break_fast['Total Carb (g)']< median_cal]
greater_df=break_fast[break_fast['Total Carb (g)']>median_cal]
plt.plot(less_df['Total Carb (g)'],marker='o',color='gold',label=f'less than {median_cal}')
plt.plot(greater_df['Total Carb (g)'],marker='o',color='cornsilk',label=f'greater than {median_cal}')
plt.title('The median values of Total Carb (g) in Burger')
plt.legend()
plt.show()

median_cal=break_fast['Dietary Fiber (g)'].median()
less_df=break_fast[break_fast['Dietary Fiber (g)']< median_cal]
greater_df=break_fast[break_fast['Dietary Fiber (g)']>median_cal]
plt.plot(less_df['Dietary Fiber (g)'],marker='o',color='royalblue',label=f'less than {median_cal}')
plt.plot(greater_df['Dietary Fiber (g)'],marker='o',color='violet',label=f'greater than {median_cal}')
plt.title('The median values of Dietary Fiber (g) in Burger')
plt.legend()
plt.show()

median_cal=break_fast['Sugars (g)'].median()
less_df=break_fast[break_fast['Sugars (g)']< median_cal]
greater_df=break_fast[break_fast['Sugars (g)']>median_cal]
plt.plot(less_df['Sugars (g)'],marker='o',color='plum',label=f'less than {median_cal}')
plt.plot(greater_df['Sugars (g)'],marker='o',color='olive',label=f'greater than {median_cal}')
plt.title('The median values of Sugars (g) in Burger')
plt.legend()
plt.show()

median_cal=break_fast['Protein (g)'].median()
less_df=break_fast[break_fast['Protein (g)']< median_cal]
greater_df=break_fast[break_fast['Protein (g)']>median_cal]
plt.plot(less_df['Protein (g)'],marker='o',color='indigo',label=f'less than {median_cal}')
plt.plot(greater_df['Protein (g)'],marker='o',color='blueviolet',label=f'greater than {median_cal}')
plt.title('The median values of Protein (g) in Burger')
plt.legend()
plt.show()

median_cal=break_fast['Weight Watchers'].median()
less_df=break_fast[break_fast['Weight Watchers']< median_cal]
greater_df=break_fast[break_fast['Weight Watchers']>median_cal]
plt.plot(less_df['Weight Watchers'],marker='o',color='crimson',label=f'less than {median_cal}')
plt.plot(greater_df['Weight Watchers'],marker='o',color='hotpink',label=f'greater than {median_cal}')
plt.title('The median values of Weight Watchers in Burgers')
plt.legend()
plt.show()

break_fast=data[data.Category=='Chicken']
median_cal=break_fast['Calories'].median()
less_df=break_fast[break_fast['Calories']< median_cal]
greater_df=break_fast[break_fast['Calories']>median_cal]
plt.plot(less_df.Calories,marker='o',color='pink',label=f'less than {median_cal}')
plt.plot(greater_df.Calories,marker='o',color='red',label=f'greater than {median_cal}')
plt.title('The median values of Calories in Chicken')
plt.legend()
plt.show()

median_cal=break_fast['Fat Calories'].median()
less_df=break_fast[break_fast['Fat Calories']< median_cal]
greater_df=break_fast[break_fast['Fat Calories']>median_cal]
plt.plot(less_df['Fat Calories'],marker='o',color='orange',label=f'less than {median_cal}')
plt.plot(greater_df['Fat Calories'],marker='o',color='wheat',label=f'greater than {median_cal}')
plt.title('The median values of Fat Calories in Chicken')
plt.legend()
plt.show()

median_cal=break_fast['Fat (g)'].median()
less_df=break_fast[break_fast['Fat (g)']< median_cal]
greater_df=break_fast[break_fast['Fat (g)']>median_cal]
plt.plot(less_df['Fat (g)'],marker='o',color='maroon',label=f'less than {median_cal}')
plt.plot(greater_df['Fat (g)'],marker='o',color='firebrick',label=f'greater than {median_cal}')
plt.title('The median values of Fat  in Chicken')
plt.legend()
plt.show()

median_cal=break_fast['Saturated Fat (g)'].median()
less_df=break_fast[break_fast['Saturated Fat (g)']< median_cal]
greater_df=break_fast[break_fast['Saturated Fat (g)']>median_cal]
plt.plot(less_df['Saturated Fat (g)'],marker='o',color='coral',label=f'less than {median_cal}')
plt.plot(greater_df['Saturated Fat (g)'],marker='o',color='tomato',label=f'greater than {median_cal}')
plt.title('The median values of Saturated Fat (g) in chicken')
plt.legend()
plt.show()

median_cal=break_fast['Trans Fat (g)'].median()
less_df=break_fast[break_fast['Trans Fat (g)']< median_cal]
greater_df=break_fast[break_fast['Trans Fat (g)']>median_cal]
plt.plot(less_df['Trans Fat (g)'],marker='o',color='chocolate',label=f'less than {median_cal}')
plt.plot(greater_df['Trans Fat (g)'],marker='o',color='peru',label=f'greater than {median_cal}')
plt.title('The median values of Trans Fat (g) in chicken')
plt.legend()
plt.show()

median_cal=break_fast['Cholesterol (mg)'].median()
less_df=break_fast[break_fast['Cholesterol (mg)']< median_cal]
greater_df=break_fast[break_fast['Cholesterol (mg)']>median_cal]
plt.plot(less_df['Cholesterol (mg)'],marker='o',color='red',label=f'less than {median_cal}')
plt.plot(greater_df['Cholesterol (mg)'],marker='o',color='salmon',label=f'greater than {median_cal}')
plt.title('The median values of Cholesterol (mg) in chicken')
plt.legend()
plt.show()

median_cal=break_fast['Sodium (mg)'].median()
less_df=break_fast[break_fast['Sodium (mg)']< median_cal]
greater_df=break_fast[break_fast['Sodium (mg)']>median_cal]
plt.plot(less_df['Sodium (mg)'],marker='o',color='cyan',label=f'less than {median_cal}')
plt.plot(greater_df['Sodium (mg)'],marker='o',color='deepskyblue',label=f'greater than {median_cal}')
plt.title('The median values of Sodium (mg) in chicken')
plt.legend()
plt.show()

median_cal=break_fast['Total Carb (g)'].median()
less_df=break_fast[break_fast['Total Carb (g)']< median_cal]
greater_df=break_fast[break_fast['Total Carb (g)']>median_cal]
plt.plot(less_df['Total Carb (g)'],marker='o',color='gold',label=f'less than {median_cal}')
plt.plot(greater_df['Total Carb (g)'],marker='o',color='cornsilk',label=f'greater than {median_cal}')
plt.title('The median values of Total Carb (g) in chicken')
plt.legend()
plt.show()

median_cal=break_fast['Dietary Fiber (g)'].median()
less_df=break_fast[break_fast['Dietary Fiber (g)']< median_cal]
greater_df=break_fast[break_fast['Dietary Fiber (g)']>median_cal]
plt.plot(less_df['Dietary Fiber (g)'],marker='o',color='royalblue',label=f'less than {median_cal}')
plt.plot(greater_df['Dietary Fiber (g)'],marker='o',color='violet',label=f'greater than {median_cal}')
plt.title('The median values of Dietary Fiber (g) in chicken')
plt.legend()
plt.show()

median_cal=break_fast['Sugars (g)'].median()
less_df=break_fast[break_fast['Sugars (g)']< median_cal]
greater_df=break_fast[break_fast['Sugars (g)']>median_cal]
plt.plot(less_df['Sugars (g)'],marker='o',color='plum',label=f'less than {median_cal}')
plt.plot(greater_df['Sugars (g)'],marker='o',color='olive',label=f'greater than {median_cal}')
plt.title('The median values of Sugars (g) in chicken')
plt.legend()
plt.show()

median_cal=break_fast['Protein (g)'].median()
less_df=break_fast[break_fast['Protein (g)']< median_cal]
greater_df=break_fast[break_fast['Protein (g)']>median_cal]
plt.plot(less_df['Protein (g)'],marker='o',color='indigo',label=f'less than {median_cal}')
plt.plot(greater_df['Protein (g)'],marker='o',color='blueviolet',label=f'greater than {median_cal}')
plt.title('The median values of Protein (g) in chicken')
plt.legend()
plt.show()

median_cal=break_fast['Weight Watchers'].median()
less_df=break_fast[break_fast['Weight Watchers']< median_cal]
greater_df=break_fast[break_fast['Weight Watchers']>median_cal]
plt.plot(less_df['Weight Watchers'],marker='o',color='crimson',label=f'less than {median_cal}')
plt.plot(greater_df['Weight Watchers'],marker='o',color='hotpink',label=f'greater than {median_cal}')
plt.title('The median values of Weight Watchers in Chicken')
plt.legend()
plt.show()

from sklearn.preprocessing import MinMaxScaler
minmax=MinMaxScaler()
trans=minmax.fit_transform(data[column[2:]])
from sklearn.cluster import KMeans
kmens=KMeans(n_clusters=10)
label=kmens.fit_predict(trans)
data['label']=label
centroids = kmens.cluster_centers_
print(centroids)
u_labels = np.unique(label)
for i in u_labels:
    plt.scatter(data.iloc[label == i, 0], data.iloc[label == i, 1], label=i)
plt.legend()
plt.show()