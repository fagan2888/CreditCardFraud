# CreditCardFraud
Main script: creditcard.m

This repository contains a library for detecting credit card fradulent transaction using MATLAB.
Credit card data was obtained through Kaggle and the logistic regression functions were adapted from Andrew Ng's Machine Learning course.
I use undersampling of the very large, very skewed data set, and logistic regression to try to classify the fraudulent transactions.
Other minor data prep is described in the code comments.

In this particular problem, recall (minimizing false negatives) is much more important than precision (minimizing false positives).
The reasoning for this is that a false positive would likely result in a transaction being flagged, for further reveiw by an agent.
These flags may be removed upon review. A false negative results in a fraudlent transaction going through, which may be much more 
problematic.
