# Chapter 4 : The perceptron

* Learning weights for features amount to learning hyperplane classifier. (a division of space into two halves by a straight line, where one half is positive and one half is negative)
-the perceptron can be seen as finding a good linear decision boundary

## Bio-inspired learning:
Based on how much these incoming neurons are firing and how strongly it wants to fire.

-Features with zero weights are ignored. Features with positive weight are indicative of positive examples because they cause the activation to increase.
-It is good to have a non zero threshold.

# Error-driven updating: the perceptron algorithm.
-The perceptron is a classic learning algorithm for the neural model of learning.The algorithm is online Instead of considering the entire data set at the same time, it only ever looks at one example. It processes that example and then goes onto the next one. It is also error drivenm as long as it does well, it does not bother to update its parameter.

* Algoritm:

w_d = 0 for all d = 1...D // initialize weights <br>
b =0 <br>
for iter =1 ... Maxiter do <br>
for all (x,y) belong to D do: <br>
&emsp; a = sum(w_d*x_d) + b // compute activation for this example <br>
&emsp; if ya <= 0 then <br>
&emsp;&emsp;w_d = w_d + y_xd for all d =1 ...d <br>
b = b + y //update bias <br>
        end if
    end for
end for
return w0,w1,...,w_d,b

-The algorthm maintains a guess at good parameters (weights and bias) as it runs. It process one example 

* The algorithm maintains a guess at good parameters as it runs. It process one example at time. For a given example, it makes a prediction. If the prediction is correct, it does nothing. When the prediction is not correct, it change the parameter in such a way it will do better next time arround.

$$a'=\sum_{d=1}^Dw'_{d} +b'$$ <br>
$$=\sum_{d=1}^D (w_{d} +x_{d})x_{d}+(b+1)$$ 

* The particular form of update for the perceptron is quite simple. The weight w_d is increased by yx_d and the bias is increased by y. The goal of the update is to make it better over time.

* The only hyperparameter of the perceptron algorithm is MaxIter, the number of pass to make over the training data. If we make many many passes over the training data, then the algorithm is likely to overfit.

* The scale of the weight vector is irrelevant from the perspective of classification.

Up to page 46 now.