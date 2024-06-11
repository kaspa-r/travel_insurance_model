# Travel Insurance Dataset Analysis

## Foreground & Goals

This Jupyter Notebook contains analysis on customerst that have purchased a Tour & Travels Company's Travel Insurance from 2019 alongside explanatory variables of each customer. The purpose of this analysis is to attempt to create the most accurate (more on that later) model for prediction of which customers will be inclined to purchase travel insurance and which - not. The flip side of this is that this analysis will try to mimic the proper machine learning process start to finish. The purpose of this analysis is two-fold:

1. Create the most accurate model for predicting potential customers based on the data we have.
2. Do so via proper machine learning process (both for educational & examplary reasons).

## Table of Contents:
    1. EDA & Data Prep
    2. Statistical Inference
    3. Model Development
        3.1 Model Setup
        3.2 Cross Validation
        3.3 Hyperparameter Tuning
        3.4 Trying Out PCA
    4. Final Testing Of Best Models
    5. Conclusions

## Variables Used

`Age`- Age Of The Customer

`Employment Type` - The Sector In Which Customer Is Employed

`GraduateOrNot`- Whether The Customer Is College Graduate Or Not

`AnnualIncome`- The Yearly Income Of The Customer In Indian Rupees[Rounded To Nearest 50 Thousand Rupees]

`FamilyMembers`- Number Of Members In Customer's Family

`ChronicDisease`- Whether The Customer Suffers From Any Major Disease Or Conditions Like Diabetes/High BP or Asthama,etc.

`FrequentFlyer`- Derived Data Based On Customer's History Of Booking Air Tickets On Atleast 4 Different Instances In The Last 2 Years[2017-2019].

## Model Building Process Utilized

1. Base model setup & validation (using the validation dataset).
2. Cross-Validation of base-models to get more concrete performance metrics.
3. Hyperparameter Tuning via the Randomized Search cross validation methodology.
4. Testing end-states of models (using the testing dataset).
5. Creating an all-encompasing voting ensemble model via the Voting Classifier.
s
## Conclusions Made

Through visual inspection:
1. High-income-earning customers seem to be purchasing travel insurance more than their counterparts.
2. Most of the clients that we have in the dataset have not Ever Travelled Abroad. It could just be because they only fly within their own country. 
3. It seems as though clients with higher amounts of family members & with more AnnualIncome purchase Travel Insurance more.
4. Clients that have Travelled Abroad before seem to be more keen in buying Travel Insurance.
5. Not sick by Chronic Diseases people tend to not take travel insurance.

Model-related:
1. After building most typical models, the best model we are able to build has the accuracy of `0.8040201005025126`, the precision of `0.4647887323943662` & the recall of `0.9705882352941176`. Note that the target of optimization was the `Recall` score.
2. The best model utilises 4 of the best test scoring models voting in a voting classifier.

## Links

Link to the Kaggle database I got my inspiration from: https://www.kaggle.com/datasets/tejashvi14/travel-insurance-prediction-data

My Personal GitHub: https://github.com/kaspa-r
My Personal LinkedIn: https://www.linkedin.com/in/kasparas-rutkauskas/