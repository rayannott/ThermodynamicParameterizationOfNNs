def AdLaLa_func(h = 0.25, gamma=0.1,T1 = 1e-4, T2 = 1e-4):

    # Load in relevant packages
    import numpy as np
    import matplotlib.cm as cm
    import numpy.random as npr
    import matplotlib.pyplot as plt
    import torch
    import torch.nn as nn
    import torchvision.datasets as datasets
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader, Dataset

    from Optimizer_AdLaLa import AdLaLa # Loads in the opimizer file

    torch.manual_seed(2) # Set random seed

    # Modify this to select the appropriate gpu
    torch.cuda.set_device(3)
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}.")

    # Generates spiral data
    def twospirals(datapoints, spiralturns = 4, noise=.02, p = 0.5, a = 2):
        
        """
         Creates a two spiral planar dataset consisting of the 2D coordinates (X) of the datapoints 
         and the corresponding labels (Y).
         The user can set the number of datapoints (N), the number of turns of the spiral (default = 4) 
         and the noise level (default = 0.02).
        """
        
        N = int(datapoints/2)  
        
        # Generate a (N,1) array with samples from the uniform distribution over [0,1)
        t = np.random.rand(N,1)
        
        # Generate noise-free training data
        dx1 = a*(t**p)*np.cos(2*spiralturns*(t**p)*np.pi) 
        dx2 = a*(t**p)*np.sin(2*spiralturns*(t**p)*np.pi)
        
        # Add noise and stack
        X = np.vstack((np.hstack((dx1,dx2)),np.hstack((-dx1,-dx2)))) + np.random.randn( 2*N,2) * noise # Coordinates
        Y = np.hstack((np.zeros(N),np.ones(N))) # Corresponding Labels
        
        return torch.Tensor(X),torch.Tensor(Y)  # Return data + labels as torch.Tensors

    class spirals(Dataset):
    
        def __init__(self,length):
            super().__init__()
            self.length = length
            self.x,self.y = twospirals(self.length)
            
        def __len__(self):
            return len(self.y)
        
        def __getitem__(self, i):
            return self.x[i], self.y[i] 

    # Generate training data set
    Ntrain = 500                # Set the number of training data points
    datanew = spirals(Ntrain)    # Generate training data points
    batchsize = 25               # The batch size can be set here
    dataloader = DataLoader(datanew, batch_size=batchsize, shuffle = True)  # Feed the data into the dataloader.

    # Generate test data set
    Ntest = 1000  # Set the number of test data points
    xtest,ytest = twospirals(Ntest) # Create the test data

    # Send test data to the GPU
    xtest = xtest.to(device)
    ytest = ytest.to(device)

    # Function to compute accuracy with
    def accuracy_binaryclass(out,y):
        diff = np.count_nonzero(np.round(out.squeeze())-y)
        return (1-(diff/np.size(y)))*100

    # Set up a single hidden-layer perceptron, where the hidden layer has 100 nodes
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            
            self.big_pass = nn.Sequential(nn.Linear(2,100), # Input layer (2 nodes) to hidden layer (100 nodes)
                                          nn.ReLU(),        # ReLU activation
                                          nn.Linear(100,1), # Hidden layer to output
                                          nn.Sigmoid()      # Pass output through a sigmoid 
                                                            # Sigmoid is appropriate for a binary classification problem
                                         )

        def forward(self, x):
            out = self.big_pass(x)
            return out 

    num_epochs = 4000  # Set the amount of epochs
    total_steps = len(dataloader)

##############################################################
    # AdLaLa hyperparameters
    eps = 0.05
    sigma_A = 1e-2

    cgamma = np.exp(-h*gamma)
    dgamma = np.sqrt(1-np.exp(-2*h*gamma))*np.sqrt(T2)
##############################################################

    # Will store results (averaged over multiple runs in here)
    RES_train_loss_avg = []
    RES_test_loss_avg = []
    RES_test_acc_avg = []
    RES_train_acc_avg = []

    for run in range(3): # Set the amount of runs to average over
        print("run =", run)

        criterion = nn.BCELoss() # Define the loss function
        NN = Net() # Generates the neural network and initializes it using standard PyTorch initialization 
        NN = NN.to(device)  # Sends the net to the GPU

        # Define the optimizer + hyperparameters
        optimizer = AdLaLa(NN.parameters(),lr=h,eps=eps,sigA=sigma_A,T1=T1,cgamma=cgamma,dgamma=dgamma)

        RES_train_loss = []
        RES_test_loss = []
        RES_test_acc = []
        RES_train_acc = []

        # Start training the neural network
        for epoch in range(num_epochs): 
            for i,data in enumerate(dataloader): # Iterate over the training data batches
                x,y = data # Training data points x + true labels y

                # Send data to the GPU
                x = x.float().to(device)
                y = y.float().to(device)

                # Only in the very first step, we call this to generate the momenta inside the optimizer file
                if i == 0 and epoch == 0:
                    output = NN(x) # The neural network's predicted labels for the provided training data 
                    loss = criterion(output,y.unsqueeze(1)) # Compute the difference between the predicted labels (output) and the true labels (y) using the defined loss function

                    # Backpropagation and update the NN's parameters
                    optimizer.zero_grad()
                    loss.backward(retain_graph=True) # retain_graph = True allows us to do .backward again within the same epoch
                    optimizer.stepMom()

                optimizer.stepAOA()

                output = NN(x)             # The neural network's predicted labels for the provided training data 
                loss = criterion(output,y.unsqueeze(1)) # Compute the difference between the predicted labels (output) and the true labels (y) using the defined loss function

                # Backpropagation and update the NN's parameters
                optimizer.zero_grad()
                loss.backward()
                optimizer.stepB()

                # Evaluate how the neural network is doing and store the results
                if (epoch+1) % 10 == 0 and (i+1) % (Ntrain/batchsize) == 0:
                    RES_train_loss.append(loss.item()) # Store the training loss

                    # Compute the accuracy of the classifier on the training data
                    acc = accuracy_binaryclass(output.cpu().detach().numpy(),y.cpu().detach().numpy())
                    RES_train_acc.append(acc)

                    # Now look at the test data: compute loss and accuracy
                    outputtest = NN(xtest)
                    loss_test = criterion(outputtest,ytest.unsqueeze(1))
                    acc_test = accuracy_binaryclass(outputtest.cpu().detach().numpy(),ytest.cpu().detach().numpy())
                    RES_test_loss.append(loss_test.item())
                    RES_test_acc.append(acc_test)

                    if (epoch+1) % 100 == 0:
                        print(f'epoch {epoch}/{num_epochs}, step {i+1}/{total_steps}, with test loss = {loss_test.item()}')
                        print("Training accuracy",acc,"% and test accuracy",acc_test,"%")

        RES_train_loss_avg.append(RES_train_loss)
        RES_train_acc_avg.append(RES_train_acc)
        RES_test_loss_avg.append(RES_test_loss)
        RES_test_acc_avg.append(RES_test_acc)

    # Store the results in a file
    with open(f'Spirals_SHLP100nodes_AdLaLa_h_{h}_gamma_{gamma}_eps_{eps}_sigA_{sigma_A}_T1_{T1}_T2_{T2}_batchsize_{batchsize}.txt', 'w+') as f:
        f.write(f'Training loss mean: {np.mean(RES_train_loss_avg,0)}\n') 
        f.write(f'Test loss mean: {np.mean(RES_test_loss_avg,0)}\n') 
        f.write(f'Training accurary mean: {np.mean(RES_train_acc_avg,0)}\n') 
        f.write(f'Test accuracy mean: {np.mean(RES_test_acc_avg,0)}\n') 
        f.write(f'Training loss min: {np.min(RES_train_loss_avg,0)}\n') 
        f.write(f'Test loss min: {np.min(RES_test_loss_avg,0)}\n') 
        f.write(f'Training accurary min: {np.min(RES_train_acc_avg,0)}\n') 
        f.write(f'Test accuracy min: {np.min(RES_test_acc_avg,0)}\n') 
        f.write(f'Training loss max: {np.max(RES_train_loss_avg,0)}\n') 
        f.write(f'Test loss max: {np.max(RES_test_loss_avg,0)}\n') 
        f.write(f'Training accurary max: {np.max(RES_train_acc_avg,0)}\n') 
        f.write(f'Test accuracy max: {np.max(RES_test_acc_avg,0)}\n') 

