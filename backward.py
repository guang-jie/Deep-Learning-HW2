# include forward and backward model
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
from dataset import CustomImageDataset, DataLoader
from function import onehot_encoding, accurate_num, turn_to_score

# Define your neural network architecture (number of layers, number of neurons in each layer, activation functions, etc.)
input_size = 512
hidden_size = 200
output_size = 50


# Initialize weights and biases for each layer
weights = [np.random.randn(input_size, hidden_size), np.random.randn(hidden_size, output_size)]
biases = [np.zeros((1, hidden_size)), np.zeros((1, output_size))]

# Define your forward pass function
def forward(inputs):
    #print("inputs=", inputs.shape) # (100, 512)   

    hidden_layer = np.dot(inputs, weights[0]) + biases[0] # (100, 200)
    hidden_layer_activation = sigmoid(hidden_layer)
    output_layer = np.dot(hidden_layer_activation, weights[1]) + biases[1] # (100, 50)
    output_layer_activation = sigmoid(output_layer)
    
    return output_layer_activation

# Define your loss function
def loss(prediction, target):
    error = target - prediction
    return np.mean(error**2) #MSE loss

# Define your sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Define your backpropagation function
def backpropagation(input, target, learning_rate):
    # Forward pass
    hidden_layer = np.dot(input, weights[0]) + biases[0]
    hidden_layer_activation = sigmoid(hidden_layer)
    output_layer = np.dot(hidden_layer_activation, weights[1]) + biases[1]
    output_layer_activation = sigmoid(output_layer)

    # Backward pass
    error = target - output_layer_activation
    output_delta = error * sigmoid_derivative(output_layer)
    hidden_error = np.dot(output_delta, weights[1].T)
    hidden_delta = hidden_error * sigmoid_derivative(hidden_layer)

    # Update weights and biases
    weights[1] += learning_rate * np.dot(hidden_layer_activation.T, output_delta)
    biases[1] += learning_rate * np.sum(output_delta, axis=0, keepdims=True)
    weights[0] += learning_rate * np.dot(input.T, hidden_delta)
    biases[0] += learning_rate * np.sum(hidden_delta, axis=0, keepdims=True)

    return loss(output_layer_activation, target), output_layer_activation



if __name__ == "__main__":
    traindataset = CustomImageDataset("train.txt") # just an object, and you need to call him
    valdataset = CustomImageDataset("val.txt")
    testdataset = CustomImageDataset("test.txt")
    trainloader = DataLoader(traindataset, batch_size=100, shuffle=True)
    valloader = DataLoader(valdataset, batch_size=100, shuffle=False)
    testloader = DataLoader(testdataset, batch_size=100, shuffle=False)
    epoch = 10000

    
    
    for i in range(epoch):
        print("epoch=", i)

        # training
        train_accuracy = 0
        train_loss_list = []        
        for batch in trainloader:
            inputs, labels = zip(*batch)
    
            images = inputs
            images = np.array(images) # (100, 512)

            label = labels
            label = np.array(label, dtype=np.int32) # (100,)
            encoded_label = onehot_encoding(label) # (100, 50)
           
            # include forward and backward model
            loss_fn, score_vector = backpropagation(images, encoded_label, learning_rate=0.0001)          

            score_list = []
            for idx in range(len(score_vector)):
                score = turn_to_score(score_vector[idx, :])
                score_list.append(score)
            score_list = np.array(score_list) # (100,)           

            # accuracy
            accurate_batch = accurate_num(label, score_list) 
            train_accuracy = train_accuracy + accurate_batch 
            
            # loss
            train_loss_list.append(loss_fn.item())

        train_accuracy = train_accuracy / traindataset.__len__()
        train_loss = np.array(train_loss_list).mean()
        print("train_accuracy=", train_accuracy)

        
        # valid       
        valid_accuracy = 0
        valid_loss_list = []
        for batch in valloader:
            inputs, labels = zip(*batch)
    
            images = inputs
            images = np.array(images) # (100, 512)

            label = labels
            label = np.array(label, dtype=np.int32) # (100,)
            encoded_label = onehot_encoding(label) # (100, 50)
           
            # forward model
            score_vector = forward(images)          

            score_list = []
            for idx in range(len(score_vector)):
                score = turn_to_score(score_vector[idx, :])
                score_list.append(score)
            score_list = np.array(score_list) # (100,)           

            # accuracy
            accurate_batch = accurate_num(label, score_list) 
            valid_accuracy = valid_accuracy + accurate_batch                        

        valid_accuracy = valid_accuracy / valdataset.__len__()        
        print("valid_accuracy=", valid_accuracy)


        # test   
        test_accuracy = 0
        test_loss_list = []
        for batch in testloader:
            inputs, labels = zip(*batch)
    
            images = inputs
            images = np.array(images) # (100, 512)

            label = labels
            label = np.array(label, dtype=np.int32) # (100,)
            encoded_label = onehot_encoding(label) # (100, 50)
           
            # forward model
            score_vector = forward(images)          

            score_list = []
            for idx in range(len(score_vector)):
                score = turn_to_score(score_vector[idx, :])
                score_list.append(score)
            score_list = np.array(score_list) # (100,)           

            # accuracy
            accurate_batch = accurate_num(label, score_list) 
            test_accuracy = test_accuracy + accurate_batch                        

        test_accuracy = test_accuracy / testdataset.__len__()        
        print("test_accuracy=", test_accuracy)

        '''
        # without feature extraction, to ensure resize function
        image = np.array(image)
        image = image.squeeze()
        print(image.shape)
        cv2.imshow(winname='image', mat=image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''
        
        
    #print(traindataset.__len__())