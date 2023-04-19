from model import LeNet5
from preprocess import get_batch, MakeOneHot
from function import CrossEntropyLoss, onehot_encoding, SGDMomentum, draw_losses, accurate_num, turn_to_score
from dataset import CustomImageDataset, DataLoader
import numpy as np
from config import batch_size, current_path
from os.path import join

model = LeNet5()
criterion = CrossEntropyLoss()
optim = SGDMomentum(model.get_params(), lr=0.001, momentum=0.80, reg=0.00003)

traindataset = CustomImageDataset("train.txt") # just an object, and you need to call him
valdataset = CustomImageDataset("val.txt")
testdataset = CustomImageDataset("test.txt")
trainloader = DataLoader(traindataset, batch_size=batch_size, shuffle=True)
valloader = DataLoader(valdataset, batch_size=batch_size, shuffle=False)
testloader = DataLoader(testdataset, batch_size=batch_size, shuffle=False)

print("traindataset =", traindataset.__len__())
print("valdataset =", valdataset.__len__())
print("testdataset =", testdataset.__len__())

# TRAIN
ITER = 100
for i in range(ITER):

    print("epoch=", i)
    # training
    train_accuracy = 0
    train_loss_list = []
    train_count = 0
    for batch in trainloader:      
        inputs, labels = zip(*batch)

        X_batch = inputs
        X_batch = np.array(X_batch) # (100, 200, 200, 3)
        b, h, w, c = X_batch.shape 
        X_batch = X_batch.reshape(b, c, h, w)
        print(X_batch.shape)

        Y_batch = labels
        Y_batch = np.array(Y_batch, dtype=np.int32) # (100,)       
        Y_batch = onehot_encoding(Y_batch) # (100, 50)

        # forward, loss, backward, step
        Y_pred = model.forward(X_batch) 
        print(Y_pred.shape) # (100, 50)
        loss, dout = criterion.get(Y_pred, Y_batch) 
        model.backward(dout)
        optim.step()
        
        
        # accuracy
        label = labels
        score_list = []
        for idx in range(len(Y_pred)):
            score = turn_to_score(Y_pred[idx, :])
            score_list.append(score)
        score_list = np.array(score_list) # (100,)
        
        accurate_batch = accurate_num(score_list, label) # original label
        train_accuracy = train_accuracy + accurate_batch
        
        
        train_count = train_count + 1
        if train_count > 10:
            break
        
        
    train_accuracy = train_accuracy / 2000
    #print(type(train_accuracy))
    train_accuracy = train_accuracy*100
    print("train_accuracy=", train_accuracy)
    
    
    # valid       
    valid_accuracy = 0
    valid_loss_list = []
    for batch in valloader:
        inputs, labels = zip(*batch)

        X_batch = inputs
        X_batch = np.array(X_batch) # (100, 200, 200, 3)
        b, h, w, c = X_batch.shape 
        X_batch = X_batch.reshape(b, c, h, w)
        #print(X_batch.shape)

        Y_batch = labels
        Y_batch = np.array(Y_batch, dtype=np.int32) # (100,)       
        Y_batch = onehot_encoding(Y_batch) # (100, 50)

        # forward, loss, backward, step
        Y_pred = model.forward(X_batch) # you should modify the model framework to fit the dataset
        #print(Y_pred.shape) # (100, 50)
        loss, dout = criterion.get(Y_pred, Y_batch) 
        
        
        
        # accuracy
        label = labels
        score_list = []
        for idx in range(len(Y_pred)):
            score = turn_to_score(Y_pred[idx, :])
            score_list.append(score)
        score_list = np.array(score_list) # (100,)
        
        accurate_batch = accurate_num(score_list, label) # original label
        valid_accuracy = valid_accuracy + accurate_batch
        break
        
        
    valid_accuracy = valid_accuracy / 200
    valid_accuracy = valid_accuracy*100
    print("valid_accuracy=", valid_accuracy)
    
    
    # test   
    test_accuracy = 0
    test_loss_list = []
    for batch in testloader:
        inputs, labels = zip(*batch)

        X_batch = inputs
        X_batch = np.array(X_batch) # (100, 200, 200, 3)
        b, h, w, c = X_batch.shape 
        X_batch = X_batch.reshape(b, c, h, w)
        #print(X_batch.shape)

        Y_batch = labels
        Y_batch = np.array(Y_batch, dtype=np.int32) # (100,)       
        Y_batch = onehot_encoding(Y_batch) # (100, 50)

        # forward, loss, backward, step
        Y_pred = model.forward(X_batch)
        #print(Y_pred.shape) # (100, 50)
        loss, dout = criterion.get(Y_pred, Y_batch) 
        
        
        
        # accuracy
        label = labels
        score_list = []
        for idx in range(len(Y_pred)):
            score = turn_to_score(Y_pred[idx, :])
            score_list.append(score)
        score_list = np.array(score_list) # (100,)
        
        accurate_batch = accurate_num(score_list, label) # original label
        test_accuracy = test_accuracy + accurate_batch
        
        
    test_accuracy = test_accuracy / 200
    test_accuracy = test_accuracy*100
    print("test_accuracy=", test_accuracy)
    
    
    
    
    
    acc = "train_accuracy: {}, valid_accuracy: {}, test_accuracy: {}".format(train_accuracy, valid_accuracy, test_accuracy)
    print(acc)
    with open(join(current_path, "acc_files"), "a+") as f:
        f.write(acc + '\n')


# save params
weights = model.get_params()
with open("weights.pkl","wb") as f:
	pickle.dump(weights, f)

draw_losses(losses)

