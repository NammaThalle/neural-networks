# Logistic Regression with Neural Networks mindset
This section explores the Logistic regression model keeping the mindset of Neural Network. It demonstrates the model building and prediction to determine whether entered picture is a "cat" picture or a "non-cat" picture.

#### 1. Packages
Import all the packages required.
1. __numpy__ - required for scientific computing with Python.
2. __h5py__ - to retrieve dataset stored in h5 file.
3. __matplotlib__ - to plot graphs in Python.
4. __copy__ - to use deepcopy - making a totally different object
5. __scipy__ - used to test with own picture at the end

#### 2. Loading the dataset of cat/non-cat picture
_Note This dataset is taken from [Kaggle][1] website_

[1]: https://www.kaggle.com/baners/a-logistic-regression-classifier-to-recognize-cats/data

1. Load the training dataset and extract the :
   * x_train - pixel values for each image for training the model
   * y_train - corresponding cat/non-cat value for training the model
2. Load the testing dataset and extract the :
   * x_test - pixel values for each image for testing the model
   * y_test - corresponding cat/non-cat value for testing the model's output
3. Extract the classes. (In our case "cat" and "non-cat")

#### 3. Preprocessing the data
1. Figure out the dimensions and the shape of the data 
   * m_train - number of training examples
   * m_test - number of testing examples
   * num_px - height and width of images (in our case height = width)
2. Flatten the image data of 3 RGB channels (numpx, numpx, numpx) into single vectors of shape (numpx * num_px * 3, 1)
3. Standardize (Normalize) the data

#### 4. Model Training
Model Training involves many steps
1. Initialize the weights and the bias to __zero__ value
2. Compute the sigmoid function. $sigmoid(z) = \frac{1}{1 + e^{-z}}$ for $z = w^T x + b$
    * for weights (w) and bias (b)
3. Forward Propagation of the network
   1. Get x_train
   2. Calculate Activation function: $A = \sigma(w^T X + b) = (a^{(1)}, a^{(2)}, ..., a^{(m-1)}, a^{(m)})$
   3. Calculate the Cost function: $J = -\frac{1}{m}\sum_{i=1}^{m}(y^{(i)}\log(a^{(i)})+(1-y^{(i)})\log(1-a^{(i)}))$
4. Backward Propagation of the network
   1. Find the gradients using the formulas:
   $$ \frac{\partial J}{\partial w} = \frac{1}{m}X(A-Y)^T\tag{7}$$
   $$ \frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^m (a^{(i)}-y^{(i)})\tag{8}$$
5. Optimize the network using the __gradient descent__
   1. Learn the $w$ and $b$ by minimizing the cost function $J$
   2. Update rule for optimizing the parameters is $ \theta = \theta - \alpha \text{ } d\theta$, where $\alpha$ is the learning rate
6. Build a predictor for predicting outcomes
   1. Find the Activation for prediction data set (x_pred) using formula in step 3.2
   2. if Activation > 0.5 -> cat picture (y_pred = 1)
7. Create a Model using steps 1-6

#### 5. Combining and making a whole pipeline
1. Load the Dataset
2. Preprocess the data
3. Train the model
4. Plot the learning curve