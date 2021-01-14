import os
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb

warnings.filterwarnings("ignore")

current_dir = os.getcwd()
file_path = os.path.join(current_dir, r"data_set/train/advertising.csv")

df = pd.read_csv(file_path)
print(df.info())

"""
#1: What age group does the data-set majorly consist of?
"""
sb.distplot(df['age'], bins=10, kde=True,
            hist_kws=dict(edgecolor="k", linewidth=1))
print('Oldest person was of:', df['age'].max(), 'Years')
print('Youngest person was of:', df['age'].min(), 'Years')
print('Average age was of:', df['age'].mean(), 'Years')

"""
#2: What is the income distribution in different age groups?
"""
sb.jointplot(x='age', y='area_income', color="blue", data=df)

"""
#3: Which gender has clicked more on online ads?
"""
report = df.groupby(['gender', 'click'])['click'].count().unstack()

sb.set_style('whitegrid')
sb.countplot(x='gender', hue='click', data=df, palette='bwr')

print(report)

plt.show()
