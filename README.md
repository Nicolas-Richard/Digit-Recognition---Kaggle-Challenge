# Digit Recognition Challenge

Solution to the Kaggle digit recognition challenge : www.kaggle.com/c/digit-recognizer/data

My code
-------

> - digit.py is used for testing and exploration.
> - digit-submission.py is used to build the submission file.
> - same scripts with '-pipeline' in the name works the same way but include PCA.

> **Note:** Best score : 0.97371


# My approach

There are many classifiers which seem to work well on this problem. I personnaly chose to go with KNN at first as it's the easier, in my opinion, to get some **inuition** on.

1st get going - Exploratory Analysis
------------------------------------

As a start I ran the KNN on a subset of the data. At this point I just wanted to run the classifier and start to see some results. 


To do so I rely on the training set only and don't get in the trouble to submit anything to Kaggle. The idea is simply to split the test set in a train set and a **cross-validation set**. Which I do 50 / 50.

How well am I doing ? Add metrics
----------------------------------

After I validated that everything was running smoothly I added some metrics printing in order to get an idea of how well / bad I was doing.


As I increased the size of the training set I saw a big improvement in the precision.


> - With 100 training records, F1 score around 0.5 after testing on the cross validation set.
> - With 1000 training records, F1 score around 0.78.
> - With 10000 training records, F1 score around 0.92.


Since the classifier is obviously getting better with more data, I decided to train it with all I have, the 42000 rows, and do my prediction based on that, then submit to Kaggle. This is very time an ressoruce consuming on my laptop. About 40 mins. 

> **Note:**
Doing so I received a score of 0.96557. (rank 388)

Adjusting this classfier using the regular options / tuning the hyperparameters
-------------------------------------------------------------------------------

Before moving on to other classifiers, I wanted to check the different options of KNN. To do so, I have a look at the documentation (http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html). And quickly realize that some options affect **easy to understand parameters**, so I decide to try these 3 ones. **Weights, distance, and number of neighbors considered**.

#### Weights

Once again I do some tests on small subsets of my dataset, in order to gain an understanding whether it's worth it to apply this on the whole dataset.

I came to realize that using a different weighting option would give 0.01 more on the F1 score.

> **Note:** From the documentation :
‘uniform’ : uniform weights. All points in each neighborhood are weighted equally.
‘distance’ : weight points by the inverse of their distance. in this case, closer neighbors of a query point will have a greater influence than neighbors which are further away.


I run the classifier on the whole set and submit to Kaggle : 

> **Note:**result a score of 0.96643 (rank 345, move up 35 positions!). Not a huge improvement but it's a start.

#### The distance 

What about the distance used. (I'm discarding my weighting option for this tests)

> - p = 1 : 1000, 0.77
> - p = 2 : 1000, 0.78
> - p = 3 : 1000, 0,80

will this hold with a bigger training set ?

> - p = 2 : 10000, 0,92
> - p = 3 : 10000, 0,93

it seems to hold but the improvement is less significant.

> - p = 4 : 1000, 0,79
> - p = 5 : 1000, 0,79

So it looks like p=3 could be our best bet here.

Now in order to test this hypothesis I'd like to 

> - test on the whole set using p = 3
> - test on the whole set using p = 3 and weights = 'distance'.

#### Neighbors

But all this is very time intensive. And I wonder if I could make it more efficient by reducing the number of neighbors I'm considering for my classification. I chose at the very start to consider 10. But the default, 5, may be enough.

Let's study how the number of neighbors affects the running time and the F1 score. To do so, I'll run myscript in ipython like this :
```
%run -t digit.py
```
and I look at the Wall time to give me the time elapsed for the whole run. Surprisingly, on 100 records, it's faster to use 10 neighbors than 5. Could it be that it takes more time to identify the closest 5 neighbors, than the closest 10.

(for these timings, I kept p = 3, but let the weight option to default)

> - 100 records, n = 5, Wall time:      12.12 s, F1 = 0.45
> - 100 records, n = 10, Wall time:       6.58 s, F1 = 0.36

> - 1000 records, n = 5, Wall time:      14.12 s, F1 = 0.81
> - 1000 records, n = 10, Wall time:       11.85 s, F1 = 0.80

> - 1000 records, n = 5, Wall time:      333.55 s, F1 = 0.94
> - 1000 records, n = 10, Wall time:       332.84 s, F1 = 0.93

#### Best set of options

In conclusion, regarding the timing I have to say I don't think I can reach any conclusion for now.

But it seems like the best choice of option would be :
```
KNeighborsClassifier(weights = 'distance', n_neighbors=5, p=3)
```

So I ran it, it was extremely slow (14876.46 s, about 4 hours) but worth it !


> **Note:**
score 0.97214, rank 210 (up 139). This is much more significant that the previous improvement.

Walking backwards, working on the features
------------------------------------------

Thankfully sci-kit learn pipelines makes very easy to combine features extraction and classification. (scikit-learn.org/stable/modules/pipeline.html)

#### Adding feature reduction : PCA

The motivation of using PCA here is to reduce number of features in the data while keeping as much relevant information as possible. In the data, we start with 784 dimensions for 28 x 28 pixels.

Using PCA I can specify how many features to keep.
http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA

> - 5000 records, without PCA, F1 score 0.93, Wall time:      52.49 s.
> - 5000 records, features keeped 500, F1 score 0.93, Wall time:     102.00 s.
> - 5000 records, features keeped 100, F1 score 0.94, Wall time:      35.22 s.
> - 5000 records, features keeped 50, F1 score 0.95, Wall time:      17.04 s.
> - 5000 records, features keeped 10, F1 score 0.89, Wall time:      12.09 s.

The good news is I now have a better and faster classifier, than I had using just the Knn classifier.

> 1. Will this hold true with the classifiers options ?
> 2. Will this prove scalable to 42000 records for training ?

#### 1. Adding the options

> - 5000 records, features keeped 100, F1 score 0.94
> - 5000 records, features keeped 100, with Knn options, F1 score 0,94

> - 5000 records, features keeped 50, F1 score 0.95
> - 5000 records, features keeped 50, with Knn options, F1 score 0.94

The hyperparameters I've previously identified as winners doesn't seem to go along so well with PCA

#### 2. Is the speed improvement still significant with 42000 records ?

Yes. Well it's very fast, if it's not a great way to improve my score I know PCA in this case is a great way to speed the classification. I'm keeping 50 features from the PCA now.

Training on 20000 records, Wall time:     208.08 s.

Create a submission file took: Wall time:     514.65 s.

And the result 0.97343, a small improvement of 0.00129 which allowed me to rank 195, by gaining 15 places.

The real gain here is the speed of the training, about 30 times faster.

Keeping 100 features : 963.19 s.

> **Note:**
Highest score yet : 0.97371, and move up two positions to reach 193.


#### What about adding feature normalization ?

In this case, adding normalization to the pipeline turns out to be counter productive.

1000, with normalization, F1 0.82
1000, no normalization, F1 0.84

5000, with normalization, F1 0.90
5000, no normalization, F1 0.93

10000, with normalization, F1 0.92
10000, no normalization, F1 0.95

Is this surpising ? First of all, given the nature of the dataset, each feature is already scaled between 0 for a white pixel and 255 for a black pixel. So all which is actually happening is about standard deviation. Could it be that by adjusting the stanrdad deviation I bring more noise ?



To go further
-------------

> - Here is a python solution at 96,9%  combining different classifiers https://www.kaggle.com/c/digit-recognizer/forums/t/4281/sharing-python-code-for-knn-svc-and-random-forest
> - Here a python script at 98,7% http://challengepost.com/software/kaggledigitrecognizer
> - And some research will tell you that best results are obtained with convolutional neural networks.
