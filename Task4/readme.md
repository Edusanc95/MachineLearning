###Features


- Study of correlation between features and total cases

- Feature selection


# Task4.md
# Task 4 for Machine Learning, by Eduardo Sánchez López and José Alejandro Libreros Montaño



**Table of Contents**

[TOCM]

[TOC]


## 1. Feature selection
Our data is from Iquitos 2004 – 2007
Only week deleted was the 2005-01-01, because it did not have any data. We did some changes to the dataset regarding empty spaces and strange explained in task 1 and 2. In retrospective we should have left the database as it was and used pandas internally to modify it while executing the script, that way we could modify our decisions on the go without having to redo the whole database cleaning. We will keep it in mind for future scripts.

In the study of the training data in the previous task we established that certain features are useless for the this problem. For this task, we are going to return to the original database and get what are the useless features by the decision trees. Then we are going to check with our previous studies and see how correct we were or were not.

First of all, we did the correlation between the features and our target, which is the total number of cases detected:

![](https://raw.githubusercontent.com/Edusanc95/MachineLearning/master/Task4/images/correlation-vs-target.png)


Our features do not have a very high correlation with the target, as we can see with a mere 0.2 correlation as the maximum. We also have some that do not even get to 0.05, we studied the possibility of deleting those but we decided against it in the end since the algorithm that we are going to use for this task (decision tree) chooses by itself the relevant features. We are going to get back to this point later on this document.

After all we did a cross-validation test to check what depth should out decision tree have.

![](https://raw.githubusercontent.com/Edusanc95/MachineLearning/master/Task4/images/cross-validation.png)

As we can see, we get the best results with depth = 2.

After that we compute the features relevancy with this depth.

![](https://raw.githubusercontent.com/Edusanc95/MachineLearning/master/Task4/images/features-relevancy.png)


As a note, we can see that the correlation is not very related with the relevancies, since two variables that are used for our decision tree (reanalysis_relative_humidity_percent and ndvi_se) are not the one that are the most correlated, sure they are one of the most correlated ones still, but it is important to note this for future tasks were will we have to manually delete some useless features.


This is the resulting tree.

![](https://raw.githubusercontent.com/Edusanc95/MachineLearning/master/Task4/images/tree1.png)

We tried with other depths to study them, but the overfitting got much much worse so we left it at 2.

Here is the cross-validation test to check visually if the 3 features used in our tree.


![](https://raw.githubusercontent.com/Edusanc95/MachineLearning/master/Task4/images/cross-validation-2.png)

![](https://raw.githubusercontent.com/Edusanc95/MachineLearning/master/Task4/images/cross-validation-3.png)

![](https://raw.githubusercontent.com/Edusanc95/MachineLearning/master/Task4/images/cross-validation-4.png)


Here we can see they fit the model nicely.


@Edusanc95

@josealejolibreros
