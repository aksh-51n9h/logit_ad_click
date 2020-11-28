# Ad-Click Prediction
A company wants to know the ***CTR ( Click Through Rate )*** in order to identify whether spending their money on digital advertising is worth or not.
<br><br>***CTR ( Click Through Rate )*** is the **ratio between the number of clicks advertisers receive on their ads and number of impressions.**
<br><br>Here we're going to train machine learning model that will predict ad-click.

### Dependecies:
- Python 3.8
- numpy : ```pip install numpy```
- sci-py : ```pip install scipy```
- pandas : ```pip install pandas```
- scikit-learn : ```pip install scikit-learn```
- matplotlib : ```pip install matplotlib```
- seaborn : ```pip install seaborn```
- tensorflow : ```pip install tensorflow```

#### Run the following in python shell:
- ```import numpy``` if you get no error message then installation is successful.


### Steps involved:
- [x] Importing data-set
    * [Ad-click Prediction Data-set](https://www.kaggle.com/c/avazu-ctr-prediction/data?select=test.gz)
   
- [x] Data-set analysis
    * In the data-analysis part we've figured out questions as follows:
        * #### What age group does the data-set majorly consist of?
            <br><img width="449" alt="figure_1.png" src="https://raw.githubusercontent.com/aksh-51n9h/logit_ad_click/main/images/dataset_visualization/Figure_1.png"><br>
        * #### What is the income distribution in different age groups?
            <br><img width="449" alt="figure_2.png" src="https://raw.githubusercontent.com/aksh-51n9h/logit_ad_click/main/images/dataset_visualization/Figure_2.png"><br>
        * #### Which gender has clicked more on online ads?
 
- [ ] How we are going to solve this problem?
    * We'll use binary classification (implemented using Logistic Regression) of whether a given ad on a given page will be clicked by a given user, with predictive features from the following aspects: 
        * ***Ad content and information*** (category, position, text, format, and so on) 
        * ***Page content and publisher information*** (category, context, domain, and so on) 
        * ***User information*** (age, gender, location, income, interests, search history, browsing history, device, and so on)
    
- [ ] Train Test split
- [ ] Training the model
- [ ] Testing the model accuracy
- [ ] Improving our model
- [ ] Model is ready for deployment
