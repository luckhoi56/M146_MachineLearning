10/8/2018
*What is learning?
*Views of learning:
    -Learning is the removal of remaining uncertainy
        -If we know the function was an m-of-n boolean function, we can use the training data to figure out which function it is
    -Learning requires good guess, small hypothesis class:
        -we can start with a very small class and make it bigger to make the hypothesis fits the data.
    -Our guess can be wrong.

*Function can be made linear-
    -How to find a good model from the hypothesis space.
    -The hypthothesis space contains infinite number of solutions
    -several functions consistent with the data.
    -We can do local search:
        -start with a linear threshold function
        -see how well it does
        -correct
        -repeat until it converges.

*general framework for learning:
    -set of possible instances X
    -set of possible labels Y
    -target function f: X-> Y
    -set of function hypothesis: H{h|h:X ->Y}
    -Input: training instances drawn from data generating function possible p
        L in the slide is the loss function. L(y, h(x)): we should have L as low as possible.

        Two things: we don't know what p is, but we are given the data drawn from p
-we can approximate the risk by the training error/empirical risk.
*Opimization problem:
    -A loss function:
        -measures the penalty incurred by classfier 
    -many loss function:
        -
        -
*How about the loss function:
    we cannot minimize 0-1 loss:
        -it is a combinational optimization problem: NP - hard.
    -idea: minimzing upper bound

-Why we should minimize upper bound:
    -f(h) <= g(h)
    If we can find an h function that minimize the h function, we can minimize the f function
*Challenges:
    -To eliminate this problem, minimize a new function -
*Overfitting the data:
    -Learning a tree that classifies the training data perfectly may not lead to good generalization.

*Bias vs variance:
    -training data are subsamples drawn from the true distribution
    -study every chapter well: low var, Bias
    -study only a few chapter: low bias, high var.
    -study every chapter roughly: low var, high bias.
*Overfitting the data:
    -learning a tree that classifies the training data perfectly may not lead to the tree with best generalization.
    -there may be noise from the training data
    -the algo might make decision based on very little data.

*Prevent Overfitting:
    -using a less-expressive model: linear model,
    -adding regularization:
        -promote simpler models
    -data pertubation:
        -make the model more robust
        -can be done algorithically
    -stop the optimization process earlier
        -bad in theory, good in practice.
*Develop ML algorithm:
    -step 1 : collect data
    -step 2: design algo
    -step 3: verify algo, tune hyper-parameters
        -split to train/Dev
        -use cross-validation
    -step 4: re-train the model using the best parameter on the whole data set
    -step 5: deploy model on real test data.
-For regression,, common choice is a squared loss.
-you need to know the performance on the test data, but you cannot see the test data.

*Train/ dev splits:
    train, dev: original training set
    -dev is like a mock exam
    train -> dev ->test
-large train, small dev: result on dev is not representative
-small trian, large dev: may not have enough data to train a model

*N-fold cross validation:
    -instead of single test-training split:
        split data into N-equal size parts
    -train and test N different classifiers
    -report average accuracy and std of the accuracy to choose parameters.

*In class-practice:
www.kaggle.com (ML problem)

# Lecture 4: Decision Tree
## 10/10/2018
* Key issue in ML:
Linear model: Find a function that best separates the data

* Motivation:
- 
Can we learn a function that is more flexible in terms of what it does to feature space

* Sample dataset:
    - Label for red triangle should be B: only B has the red, and the triangle
    - green color can be A or B
    - For this example, shape first and then color

* Decision tree:
    - A data structure that represents data by implementing divide and conquer
    - can be useed as a classification or regression method
    - given a collection of example, learn a decision tree that can represent it
    - use this representation to classify new example
* Note: If still cannot classify, define a new feature to classify the model

* root: top node
* leaf: node that has no further children
* edge : connection between the two nodes

* Motivation: Many decision are tree structures:
    - Medical treatment:
* Decision boundaries: handling real-valued features
    - instances are represented as attributes values

* Advantages of decision trees:
    - can represent any boolean function
    - can be viewed as a way to represent a lot of data
    - natural representation
    - the evaluation of the decision tree classifer is easy

* Any boolean function can be represented by the decision tree
    - output is discrete category. real value output 

* to build deision tree, we need to know:
    - pick one best represent feature
    - values for the leaves = the output label. When to stop? 

* Algoo:
    If all examples are labeled the same
        return a single node tree with the label
    Otherwise
        recursive traverse the decision algo

* ID3 Algo:
If all examples have a same label //label
    return a single node tree with a label
create a root node for a tree
    A = Attribute in Attributes that best classifies S
for (each value v of A)
    add a new tree branch corrsponding to A =v
    Let Sv be the subset of examples in S with A =v
    if Sv is empty:
        add leaf node with the common value of Label in S
    else:
        below this branch add the subtree
        ID3 (Sv attributes -{a}, label)
return root


* How do we choose which attribute to split?
    - If the splitting does not produce too much overlapse within a subset, it is a good attribute
* How to measure information gain?
    - Idea: gaining information reduces entropy
    - uncertainty can be measured by entropy
* Entropy:
* If a random variable S has K different values a1, a2,...ak its entropy is given by:

$$H[s]=\sum_{v=1}^K P(S=a_{v}log_2P(S=a_v)$$

* Measures the amount of uncertainty of a RV with a specific distribution. Higher it is, less confident for the outcome

* If we only have 2 labels (special case of the entropy):

- In average, how many bits do we need to send the msgd
- if we have only 4 tokens (A,B,C,D), how do we encode them
- Ex: All example belong to the same category:
    -aaaaaaaaaa
        - no need to communicate
- if all the examples are equally mixed (0.5 0.5)
    - two bits for each token (a:00, b:01, c:10, d:11)

scheme : a is 10, b is 11, c is 0

probability:
a: 1/4  2
b: 1/4  2
c: 1/2  1

entropy = 1/4 *2 + 1/4 *2 + 1/2 *1 = 3/2


* Information Gain:
    -The information gain of an attribute a is the expected reductiion in e

* Information Gain example outlook:
    -   For rain: H_r= -(3/14 * log(3/14) + 2/14 *log(2/14)) = 0.673 (wrong asnwer double check)
