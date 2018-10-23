10/22/2018
* Logistic Regression:
Midterm on 11/5
- Hw 2: math part 10/31
        Programming part : 11/10
Perceptron cannot be used with non-linear separable.
* Probabilistic model:

* Online learning:
    - initialize a model:
    Loop:
        given one data point
        update the model
    return the final model

* Batch Learning vs Online learning:
    Give a training set D
    Train the model
    Predict on the test set D'
Note: we can use online learning on the batch setting

* Batch vs learning:
- batch will be faster in terms of the update since batch can see many examples at a time.
- for batch we can see the overview of geometry of the data (which is better).

- for online learning, if we already have the trained model, we can do the update without replace the whole new model.
- we don't have to wait a long time to get the data. Even it is bad to just use one point at a time, we can go back an train later once we have more data.

* Different learning protocols:
    - The assumption and learning goal may be different.
    - Batch learning assumes data are independent and identicallly distributed. While online learning may provide worst-case bound under advesarial data.

Computation:
    - Space: online learning consider one instance at a time
    - Convergence rate: some fast converged optimization method require access to the entire dataset.

* Perceptron is an online learning algorithm.

* Logistcic regression:
    How to model - hypothesis space, model, definition

* The Setting:
    - Binary classification
    - inputs: feature vector where x belongs to R^N

* Linear threshold units(LTUs) classify an example 


* Original problem change from (-1, 1) to (0-1)

* Predict the gender by the height.

So how do we design this transformation:
    - Idea 1: function always output positive value
    - exp is always positive
    - exp (w^T X + b) akwasy return positve value

    - Idea 2: normalize  the value such that it is less than one
    -let us sogmoid function fo normalization

    sigmoi(z) = exp(z) / (1+ exp(z))
            = 1/ (1+exp(-z))

- What is derivative with respect to z?
d/dz(sigmoi)
Let k = 1 + exp(-z)
dk/dz = -exp(-z)
d/dz (sigmoid) = d/dk * dk/dz = (1+exp(-z))^-2*(exp(-z)) = sigmoi (z)[1-sigmoi(z)]

- What is the hypothesis space?
    -    What are the input/output
        - the input and output may be different from your model
        - features selection may remove some feature from the data
    - What is the hypothesis space
    - How 
    - If this is greater than half, predict one else predict -1
* Correspond to terms of w^Tx?
    1/ (1+e^-wt ) > 1/2  -> w^Tx >0

* we need to do this way because we want to have linearly separable function. That is why we need to map it.

* Logistic Regresssion: Setup
    - Training data:

* Maximum likelihood:
    Which bags of words morel likely generate
        - The one  on the left is more likely. The probability when you pick the letter on the left is higher.

* Purple and yellow:
the joint probability: theta ^k (1-theta) ^(n-k)
What is the best p making the joint probability maximal?
We take the log of the joint probability:




