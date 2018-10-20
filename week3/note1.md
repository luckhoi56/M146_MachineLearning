* Training data partitions

* Reca

* Questions: How the boundaries are different in KNN and decision tree?

* Useful for pattern regconition

* Motivation: Learning from memorization Recognizing flowers:
    - what is the flower called?
        - Types of Iris: setosa, versicolor, and virginica
        - Features: the widths and lengths of sepal and petal
        - we can plot the data to do the data analysis

* Labeling an unknown flower
* Closer to the red cluster
    -> label

* What will you learn in this lecture:
    - What is the KNN algorithm?

* Nearest neighbor: The basic version
    - Traning example are vectors xi, associated with a label y_i
        - x_i = a feature vector for an email y_i = SPAM
* Learning: Just store all the training examples
* Prediction: for a new example x
    - Find the k closest training examples to x
    - Cosntruct the label of x using these k points
        - For classification: every neighbor votes on the label. Predict the most frequent label among the neighbors
        - For regression: predicts the mean value

*How do we measures the distances between instaces?

* Distance between instaces:
    - euclidian distances: ||x1-x2|| = sqrt(sum(x1,i -x2,i)^2)
    - manhattance distace: ||x1-x2|| = |sum (x1,i- x2,i)|

* Distance between instances:
    - Symbolic / categorical features:
        - Most common distance is the hamming distance:
            - number of bits that are different
            - or number of features that have a different value
            - example:
                -x_1:{shape = triang;e. clolor = red. location = left, orientation: up}

                [
                    color: either 1 or 0
                    shape: either 1 or 0
                ]
                - x_2: {shape = triangle. color = blue, location: left, orientation: }
                - if some features are not important than the other, we can add the weight to the feature depends on our decision

- to decide good distance matrix, we also need to look at the data distribution, sometimes they are not a simple straightline that we think.

* Hyper-parameters in KNN:
    - choosing K (nearest neighbors)
    - distance measurement (eg in the L_p norm)

-Those are not specified by the algorithm itself
    - requires emperical studies
    -
- Training data (set)
    - N samples/ instances: D_train = {(x1, y1), (x2, y2), ....,(xn, yn)}
-Test evaluatuon data:
    - M samples /instances: D_test = {(x1, y1), (x2,y2,...,(xn, yn))}
    - they are used for assing how well h).) will do in predicting an unseen x ( x does not belong to test data)

* Recipe of train / dev /test
* For teach possible value of the hyper-parameter:
    - train model using D_train
    - evaluate the performance on the D_dev
* Choose the model parameter with the best performance on D_dev
*(optional) re-train the model on D_train v U _dev with the best parameter set
* Evaluate the model on D_test

* N-fold cross validation:
    - split the data into N equal size

* Put together:
    - training: store the entire training set
    - test:


*Other tricks:
    - use an odd K (why? to break ties)
    - neighbor labels could be weighted by their distance
        - related to kernel method
        - feature normalization could be important

* Normalize data to have zero mean and unit standard deviation in each dimension
* scale the feature accordingly.

* The decision boundary for KNN:
    - Is the K-nearest neighbor algo explicitly building a function?

* The voronoi diagram:
    - For any point x in a training set S
* voronoi digram of training exaples

* Decision boundary of 1-NN:
    - What about K-nearest neighbors?
        - Also partitions the space, but much more complex decision boundary.

* Instance based learning:
    - A class of learning methods:
        - learning: storing examples with labels
        - prediction: when presented a new example, classfify the labels using similar stored examples.
    
* K-nearest neighbors algo is an example of this class of methods
* Most of the computation is performed at prediction time:
    - open book vs closed book exams

* Advantages:
    - 

* Disadvantages:
    - need a lot of storage
    - prediction can be slow
    - naively: O(dN) for N training examples in d dimensions
    - more data will make it slower

* nearest neighbors are fooled by irrelevant attributes 
    * important and subtle
* curse of dimensionality:


* Suppose we have 1000 dimensional feature vectors:
    but only 10 features are relevant
    - distance will be dominated by 

* The curse of dimensionality:
    - What fraction of the points in cube lie outside the sphere described in it?
    - What fraction of the square is outside the inscribed circle in 2 dimensions?

* Indication:
   - in high dimensions, most of the volume of the cube is far away from the center

   * To sumnary, for k nearest neighbor, we want low dimnesion

   * we can use dimension reduction method to make the problem simpler
    
* Linear Classification:
    - Classification: use an object characteristics to know which class/category it belongs to
    - Classification is an example of pattern recognition 

- linear classfication algorithm (classifier) that makes its classification based on a linear predictor function combining a set of weights with the feature vector
- Decision boundary is flat
- may involve non-linear operations

* Different approaches:
    -explicitly creating the discriminant function 
        - perceptron 
        - support vector machine

* Discriminant function:
    - Two classes: y(x) = w^T * X + w_o

* Discriminant functions least squares
    - simultaneouslt fit a linear model to each of the columns of Y.
        -weights will have a close form W = (X^T*X)^-1 *X^T*Y

* Classify a new observation x:
    - for each class calculuate the f(x) = W.X
    - select the class with higher value f(x)

* discriminant functions least squares
    - works well
    - linearly separable
    - few outliers
    - K = 2

* Learning by splitting input space:
    - Learning by memorization

* Today: Perceptron
    - Learning by mistakes
* Recap: X as a vector space
    - X is an N-dimensional vector space (R^N)
        - each dimension = one feature
    - each x is a feature vector

* Linear threshold unit: classify an example x using the following classification rule
    output = sgn (w^T*x +b) = sgn (b + sigma(w_i, x_i))

- linear threshold units (LTUs) classify an example x using the following classfication rule

* Hypothesis space: linear model
    How do we find a correct line? (also given that many lines to choose from)

-w^T*x +b hyper-plane is also called linear function

* The Perceptron algorithm:
    - the goal is to find a separating hyperplane
        - for separable data, guaranteed to find one
    - An online algo:
        -process one example at a time
    converges if data is separable
        --mistake bound
    - several variants exists

* Suppose we have made a mistake on positive example:
    - that is y+=1 and w_t^T*x <=0

* Checkpoint: 
    - The perceptron algorithm
    - the geometry of the update
    - what can it represent

* Convergence theorem:
    - If there exists a set of weight that are consistent with the data (given the data is linearly separable), the perceptron algo will converge
* Cycling theorem:

* Margin:
    - The margin of a hyperplane for a dataset is the distance between the hyperplane and data point nearest to it

* Mistake bound theorem:
    gamma: if we have 2 groups of point, we assume the two groups can be separated.

* Suppose we have a binary classfication dataset with n dimensional inputs. If the dataset is separable, then the perceptron algo will make at most (R/ gamma)^2 mistakes on the training sequence.

- The data has a margin gamma
- the data is separable
- gamma is the complexity parameter that defines the separability of the data.

Proof (preliminaries)
* The setting:
    * Initial weigh vector w is all zeros
    * All training examples are contained in a ball of size R:
    
* Perceptron algorithm can stop at (R/gamma)^2, wrong. # of mistakes is not # of data points we see
    - perceptrons makes no assumption about data distribution.
    - after a fixed number of mistake, we are done

* but real world its not linearly separable
    - we can add more features, try to be linearly separable if possible.

* Variants of perceptron:
    - Voting and averaging:
        - aggregating the models on the learning path may give better results
            -especially when the data is not separable

* Voted perceptron:
    - remmeber every weight vector in your sequence of updates.
    - at final prediction time, each weight vector gets to vote on the label
    - The number of votes it gets is the number of iterations it survived before being updated.
    - comes with strong "theoretical" guaranteess about generalization, but impractical due to storage issues. 
* Average perceprtron: sort of like combining and take avarage on the other perceptron algorithm

* Marginal perceptron:
    - Make updates only when the prediction is incorrect
    - if the prediction is close to being incorrect, pick a small positive epsilon and update when y_i*w^T*x_o <= epsilon
    