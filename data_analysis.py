import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb

sb.set_theme()

n = 1000

n_rows = n
df = pd.read_csv(r"F:\PycharmProjects\logit_ad_click\data_set\train\advertising.csv", nrows=n_rows)

"""
#1: What age group does the data-set majorly consist of?
"""
sb.distplot(df['age'], bins=20, kde=True, hist_kws=dict(edgecolor="k", linewidth=1))

"""
#2: What is the income distribution in different age groups?
"""
sb.jointplot(x='age', y='area_income', color="green", data=df)

"""
#3: Which gender has clicked more on online ads?
"""
report = df.groupby(['gender', 'click'])['click'].count().unstack()
print(report)
plt.show()
