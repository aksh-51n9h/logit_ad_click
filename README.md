# Ad-Click Prediction
Machine learning in has evolved in recent time. It has now a variety of applications in the real world such as predictive analysis, automated things, decision making for business purpose.

We can apply this knowledge of machine learning into the business purpose for making a profit, and also discovering something interesting from the data.

## Motivation 
The incentive behind this project is "CTR (Click Through Rate)". In simple words, CTR is the ratio between the number of clicks advertisers receive on their ads and the number of impressions.

A company wants to know the **CTR ( Click Through Rate )** to identify whether spending their money on online marketing is worth or not.

If CTR is higher that means users are more interested in the specific campaign, else wise if CTR is lower then it means that the ad may not be relevant to the users.

## Problem Statement
The goal of this project is to predict if a user would click on ad based on the features of the user.

Few assumptions are as follows:
1) There male and female ratio is almost the same.
2) The age group in the data-set is between 19 to 61.
3) Ad topics are not taken into consideration.

#### Dependecies:
- Python 3.8
- numpy : ```pip install numpy```
- sci-py : ```pip install scipy```
- pandas : ```pip install pandas```
- scikit-learn : ```pip install scikit-learn```
- matplotlib : ```pip install matplotlib```
- seaborn : ```pip install seaborn```
- tensorflow : ```pip install tensorflow```

## Challenges Faced:
Before working on the dataset, there are a few challenges in this study:
1. Some amount of data is made public for ad click analysis.
2. Using a machine learning model, some challenges can be tackle down as companies will be able to target an ad to a particular set of users.


## Data-set description
The data-set consist of the following features:
- **Daily Time Spent on Site:** Consumer time on site in minutes.

- **Age:** Customer age in years.

- **Area Income:** Avg. Income of geographical area of consumer.

- **Daily Internet Usage:** Avg. minutes a day consumer is on the internet.

- **Ad Topic Line:** Headline of the advertisement.

- **City:** City of consumer.

- **Gender:** Gender of the consumer.

- **Country:** Country of consumer.

## Cleaning data-set:
- Dropped 'Advertisement Topic Line' as mentioned above. Since if we want to extract some useful then we've to apply NLP and for now, it is not part of this project.

## Data Analysis:
In the data-analysis part we've figured out some interesting questions and tried to find  answers for the same.

### What age group does the data-set majorly consist of?
<br><img width="480" alt="figure_1.png" src="https://raw.githubusercontent.com/aksh-51n9h/logit_ad_click/main/images/dataset_visualization/Figure_1.png"><br>
From the above graph, we can observe that the oldest person in the data-set is 61 years old and the youngest person is 19 years old.
<br><img width="480" alt="age_fig.png" src="https://raw.githubusercontent.com/aksh-51n9h/logit_ad_click/main/images/dataset_visualization/age_fig.png"><br>


### What is the income distribution in different age groups?
<br><img width="480" alt="figure_2.png" src="https://raw.githubusercontent.com/aksh-51n9h/logit_ad_click/main/images/dataset_visualization/Figure_2.png"><br>
From the above graph, we can observe that earnings of the age group of 25-35 are higher(56K - 70K).

### Which gender has clicked more on online ads?

## Techniques Used:

## Decision Tree:
Decision Trees are easy to understand, and easy to debug. Decision trees implicitly peform feature selection non-linear realtionships between features.

### Accuracy score from Decision Tree:
<br><img width="480" alt="auc_score_dt.png" src="https://raw.githubusercontent.com/aksh-51n9h/logit_ad_click/main/images/auc_score/auc_score_dt.png"><br>

## Logistic Regression:
Logistic Regression implementation is like walk in the park and efficient to train.

### Accuracy score from Logistic Regression:
<br><img width="480" alt="auc_score_lr.png" src="https://raw.githubusercontent.com/aksh-51n9h/logit_ad_click/main/images/auc_score/auc_score_lr.png"><br>

***We've implemented both the techniques from scratch for better understanding of ours.***

## Comparision between Decision Tree and Logistic Regression:
