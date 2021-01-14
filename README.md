# Logit Ad-Click
Online advertising is a multi-billion-dollar business producing most of the revenue for search engines. Recently one area is drawing attention among both researchers and machine learning practitioners i.e., Click Through Rate. Being able to predict this rate means meeting business targets. Accurate Click Through Rate (CTR) prediction can not only improve the advertisement company’s reputation and revenue, but also help the advertisers to optimize the advertising performance. In this project, we will develop a CTR prediction approach by utilising real time advertising data by the following strategies: user profile system is created for the purpose of classifying the advertisement data. The output of this user graph system includes the user’s age, gender, and the interest preferences. We will develop CTR prediction model by Decision Tree Model and Logistic Regression and compare them.

## Click-through Rate (CTR)
Click-through rate (CTR) is defined as the ratio of users who click on a specific ad to the number of total users who view a page, email, or advertisement. It is used to measure the success of an online advertising campaign for a particular website as well as the effectiveness of email campaigns.

## Methodology
To implement the above goals, the following methodology need to be followed:
1.	Analysis of dataset.
2.	Training of models.
3.	Model improvement.
4.	Final Model.


## Problem Statement
The goal of this project is to predict if a user would click on ad based on the features of the user.

Few assumptions are as follows:
1) There male and female ratio is almost the same.
2) The age group in the data-set is between 19 to 61.
3) Ad topics are not taken into consideration.

#### Dependencies:
- Python 3.8
- numpy
- sci-py
- pandas
- scikit-learn
- matplotlib
- seaborn
- tensorflow

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

## Dataset Analysis:
In the data-analysis part we've figured out some interesting questions and tried to find  answers for the same.

### What age group does dataset consists of?
<br><img width="480" alt="figure_1.png" src="https://raw.githubusercontent.com/aksh-51n9h/logit_ad_click/main/images/dataset_visualization/Figure_1.png"><br>
From the above graph, we can observe that the oldest person in the data-set is 61 years old and the youngest person is 19 years old.
<br><img width="480" alt="age_fig.png" src="https://raw.githubusercontent.com/aksh-51n9h/logit_ad_click/main/images/dataset_visualization/age_fig.png"><br>


### What is the distribution of annual income in different age groups?
<br><img width="480" alt="figure_2.png" src="https://raw.githubusercontent.com/aksh-51n9h/logit_ad_click/main/images/dataset_visualization/Figure_2.png"><br>
From the above graph, we can observe that earnings of the age group of 25-35 are higher(56K - 70K).

### Which gender has clicked more on online ads?
<br><img width="480" alt="figure_3.png" src="https://raw.githubusercontent.com/aksh-51n9h/logit_ad_click/main/images/dataset_visualization/Figure_3.png"><br>

## Techniques Used:

## Decision Tree:
Decision Trees are easy to understand, and easy to debug. Decision trees implicitly perform feature selection non-linear relationships between features.

### Accuracy score from Decision Tree:
<br><img width="480" alt="auc_score_dt.png" src="https://raw.githubusercontent.com/aksh-51n9h/logit_ad_click/main/images/auc_score/auc_score_dt.png"><br>

## Logistic Regression:
Logistic Regression implementation is like walk in the park and efficient to train.

### Accuracy score from Logistic Regression:
- Accuracy score with gradient descent:
<br><img width="480" alt="auc_score_lr.png" src="https://raw.githubusercontent.com/aksh-51n9h/logit_ad_click/main/images/auc_score/auc_score_lr_gd.png"><br>
- Accuracy score with stochastic gradient descent:
<br><img width="480" alt="auc_score_lr_sgd.png" src="https://raw.githubusercontent.com/aksh-51n9h/logit_ad_click/main/images/auc_score/auc_score_lr_sgd.png"><br>

***We've implemented both the techniques from scratch for better understanding of ours.***

## Comparison between Decision Tree and Logistic Regression:
From our training results we can conclude that Logistic Regression with stochastic gradient descent optimizer has more accuracy for determining the ad click probability.

## Future Scope
Having full access to the original dataset would be interesting. Explore and manipulate features would undoubtedly bring us exciting and robust insights to optimize model performance. Thus, the feature selecting step would be highly accurate. Deal with imbalanced data is considered a real-world data science problem. Exploring further methods would be interesting. The deep neural network may be an interesting for the further future study of the CTR prediction.

## References
1.	SEN ZHANG, QIANG FU, WENDONG XIAO (2017). Advertisement Click-Through Rate Prediction Based on the Weighted-ELM and Adaboost Algorithm. Scientific Programming, vol. 2017, Article ID 2938369. https://www.hindawi.com/journals/sp/2017/2938369/.
2.	PRASHANT GUPTA (2017). Decision Trees in Machine Learning. https://towardsdatascience.com/decision-trees-in-machine-learning-641b9c4e8052 .
3.	AYUSH PANT (2019). Introduction to Logistic Regression. https://towardsdatascience.com/introduction-to-logistic-regression-66248243c148 .
4.	BADREESH SHETTY (2018). Natural Language Processing(NLP) for Machine Learning. https://towardsdatascience.com/natural-language-processing-nlp-for-machine-learning-d44498845d5b
5.	AURELIEN GERON (2017). Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems.
6.	EDWARD LOPER, EWAN KLEIN, AND STEVEN BIRD (2009). Natural Language Processing with Python.
