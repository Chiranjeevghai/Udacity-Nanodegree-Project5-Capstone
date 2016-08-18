# Capstone-project

Steps to run the project 

1. Download the training data from URL.
    http://ufldl.stanford.edu/housenumbers/train_32x32.mat

2. Download the testing data from URL.
    http://ufldl.stanford.edu/housenumbers/test_32x32.mat

3. Install tensorflow. 
    https://www.tensorflow.org/versions/r0.10/get_started/os_setup.html

4. Replace the path of training and testing data at line.

    trainData = scp.loadmat('../data/train_32x32.mat')
    testData=scp.loadmat('../data/test_32x32.mat')

5. Run the code.

6. You can see the graphs at Tensorboard using command. 

    tensorboard --logdir=/tmp/mnist_logs4
    at url http://localhost:6006/