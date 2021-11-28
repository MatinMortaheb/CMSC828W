#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 01:04:45 2020

@author: matinmortaheb
"""

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np
from matplotlib import pyplot as plt

from torch.utils.data import SubsetRandomSampler

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc3 = nn.Linear(784,50)
        self.fc4 = nn.Linear(50,10)

    def forward(self, x):
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        output = F.log_softmax(x, dim=1)
        return output

def TopK_chooser(data,k, freq):
    topk_vector = torch.zeros(data.shape[0])
    #for i in range(data.shape[0]):
    (top_values,top_indeces) = torch.topk(abs(data),k)
    topk_vector[top_indeces]=data[top_indeces]
    freq[top_indeces] = freq[top_indeces] + 1
    
    return topk_vector, top_indeces

def train(args, model, device, train_loader, optimizer, epoch, k, freq, k2,n_worker=1):
    model.train()
    criterion = nn.CrossEntropyLoss()
    accumulated_comp_grad = torch.zeros([39760])
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = data.view(data.shape[0],data.shape[1]*data.shape[2]*data.shape[3])
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        grad_dataset = torch.zeros([0])
        parameter = torch.zeros([0])
        i=0
        for ps in model.parameters():
            if i%2 == 0:
                grad_teta = ps.grad
                grad_teta = grad_teta.view(grad_teta.shape[0]*grad_teta.shape[1])
                par = ps.view(ps.shape[0]*ps.shape[1])
                grad_dataset = torch.cat((grad_dataset, grad_teta), 0)
                parameter = torch.cat((parameter, par), 0)
            i += 1
        i=0
        for ps in model.parameters():
            if i%2 == 1:
                grad_teta = ps.grad
                #grad_teta = grad_teta.view(grad_teta.shape[0]*grad_teta.shape[1])
                grad_dataset = torch.cat((grad_dataset, grad_teta), 0)
                parameter = torch.cat((parameter, ps), 0)
            i += 1
        """
        if epoch == 1 and batch_idx<50:
            compressed_grad, top_indeces = TopK_chooser(grad_dataset,k, freq)
        else:
            if epoch == 1 and batch_idx == 50:
                pruned_parameters = torch.zeros([39760])
                pruned_parameters[top_indeces] = parameter[top_indeces]
                [w1,wprime] = pruned_parameters.split(39200)
                w1 = w1.reshape([50,784])
                [w2,wprime2] = wprime.split(500)
                w2 = w2.reshape([10,50])
                [w3,w4] = wprime2.split(50)
                #model.state_dict()['fc3.weight'] = w1
                #model.state_dict()['fc4.weight'] = w2
                model.fc3.weight.data = w1.clone();
                model.fc4.weight.data = w2.clone();
                model.fc3.bias.data = w3.clone();
                model.fc4.bias.data = w4.clone();
        """        
        _, top_indeces = TopK_chooser(grad_dataset,k2, freq)
        accumulated_comp_grad += grad_dataset
        
        #[w1,wprime] = compressed_grad.split(39200)
        #w1 = w1.reshape([50,784])
        #[w2,wprime2] = wprime.split(500)
       # w2 = w2.reshape([10,50])
        #[w3,w4] = wprime2.split(50)
        #model.fc3.weight.grad = w1.clone();
        #model.fc4.weight.grad = w2.clone();
        #model.fc3.bias.grad = w3.clone();
        #model.fc4.bias.grad = w4.clone();

        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()), "for worker ", n_worker)
            if args.dry_run:
                break
    return accumulated_comp_grad, top_indeces




def train1(args, model, device, train_loader, optimizer, epoch, k, freq, k2,n_worker,mask_indices):
    model.train()
    criterion = nn.CrossEntropyLoss()
    accumulated_comp_grad = torch.zeros([39760])
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = data.view(data.shape[0],data.shape[1]*data.shape[2]*data.shape[3])
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        grad_dataset = torch.zeros([0])
        parameter = torch.zeros([0])
        i=0
        for ps in model.parameters():
            if i%2 == 0:
                grad_teta = ps.grad
                grad_teta = grad_teta.view(grad_teta.shape[0]*grad_teta.shape[1])
                par = ps.view(ps.shape[0]*ps.shape[1])
                grad_dataset = torch.cat((grad_dataset, grad_teta), 0)
                parameter = torch.cat((parameter, par), 0)
            i += 1
        i=0
        for ps in model.parameters():
            if i%2 == 1:
                grad_teta = ps.grad
                #grad_teta = grad_teta.view(grad_teta.shape[0]*grad_teta.shape[1])
                grad_dataset = torch.cat((grad_dataset, grad_teta), 0)
                parameter = torch.cat((parameter, ps), 0)
            i += 1
        """
        if epoch == 1 and batch_idx<50:
            compressed_grad, top_indeces = TopK_chooser(grad_dataset,k, freq)
        else:
            if epoch == 1 and batch_idx == 50:
                pruned_parameters = torch.zeros([39760])
                pruned_parameters[top_indeces] = parameter[top_indeces]
                [w1,wprime] = pruned_parameters.split(39200)
                w1 = w1.reshape([50,784])
                [w2,wprime2] = wprime.split(500)
                w2 = w2.reshape([10,50])
                [w3,w4] = wprime2.split(50)
                #model.state_dict()['fc3.weight'] = w1
                #model.state_dict()['fc4.weight'] = w2
                model.fc3.weight.data = w1.clone();
                model.fc4.weight.data = w2.clone();
                model.fc3.bias.data = w3.clone();
                model.fc4.bias.data = w4.clone();
        """
        masked_grad_dataset = torch.zeros([39760])
        masked_grad_dataset[mask_indices] = grad_dataset[mask_indices]        
        _, top_indeces = TopK_chooser(masked_grad_dataset,k2, freq)
        accumulated_comp_grad += masked_grad_dataset
        
        #[w1,wprime] = compressed_grad.split(39200)
        #w1 = w1.reshape([50,784])
        #[w2,wprime2] = wprime.split(500)
       # w2 = w2.reshape([10,50])
        #[w3,w4] = wprime2.split(50)
        #model.fc3.weight.grad = w1.clone();
        #model.fc4.weight.grad = w2.clone();
        #model.fc3.bias.grad = w3.clone();
        #model.fc4.bias.grad = w4.clone();

        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()), "for worker ", n_worker)
            if args.dry_run:
                break
    return accumulated_comp_grad, top_indeces


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(data.shape[0],data.shape[1]*data.shape[2]*data.shape[3])
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def take_subsetdata_indices(dataset1, num_worker, uniform=True):
    data=dataset1.data
    target=dataset1.targets
    
    num_data=len(dataset1)
    index_set=np.linspace(0,num_data-1,num_data).astype(int)
    data_perworker= int(num_data/num_worker)
    num_class_perworker=int(10/num_worker)
    class_category=np.linspace(0,9,10).astype(int)
    remainder= 10-num_class_perworker*num_worker
    indices=[]
    for i in range(num_worker):
        indices.append(list([]))
    
    if uniform:
        for i in range(num_worker):
            indices[i]= list (index_set[i*data_perworker:(i+1)*data_perworker])           
        return indices
    else:
        count=0
        for i in range(num_worker):
            for j in range(num_data):
                for k in range(num_class_perworker):
                    if(target[j]==i*num_class_perworker+k):
                        indices[i].append(j)
            if(i==num_worker-1):
                for j in range(num_data):
                    for k in range(remainder):
                        if(target[j]==i*num_class_perworker+(k+1)):
                            indices[i].append(j) 
        return indices
            


def main():
    num_worker=10
    
    
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=5000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = False #use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'batch_size': args.batch_size}
    if use_cuda:
        kwargs.update({'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': False},
                     )

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        #transforms.Normalize((0.5,), (0.5,))
        ])
    
    dataset1 = datasets.MNIST('../../mortaheb/data', train=True, download=True,
                              transform=transform)
    dataset2 = datasets.MNIST('../../mortaheb/data', train=False,
                              transform=transform)
    
    test_loader = torch.utils.data.DataLoader(dataset2, **kwargs)
    
    models=[]
    optimizers=[]
    global_model=Net().to(device)
    optimizer_global=optim.SGD(global_model.parameters(), lr=args.lr)
    freq3= np.zeros(39760)
            
    #freq= np.zeros([num_worker,k])

    for n_worker in range(num_worker):
        models.append(Net().to(device))
        models[n_worker].load_state_dict(global_model.state_dict())
        optimizers.append(optim.SGD(models[n_worker].parameters(), lr=args.lr))
        
    j=0    
    for epoch in range(1, args.epochs + 1):
        average_grad=torch.zeros([39760])
        #plt.subplots(nrows=10, ncols=1, sharex='all', sharey='all')
        for n_worker in range(num_worker):

            subset_indices= take_subsetdata_indices(dataset1, num_worker, uniform=True)
            sampler=SubsetRandomSampler(subset_indices[n_worker])
            train_loader = torch.utils.data.DataLoader(dataset1,**kwargs,sampler=sampler)
            
            k = 39760#4000
            k2 = 39760
            k3 = 40
            
            #freq= np.zeros([num_worker,k])
            freq= np.zeros(39760)
            freq2= np.zeros(39760)
    
            parameter = torch.zeros([0])
            i=0
            for ps in global_model.parameters():
                if i%2 == 0:
                    par = ps.view(ps.shape[0]*ps.shape[1])
                    parameter = torch.cat((parameter, par), 0)
                i += 1
            i=0
            for ps in global_model.parameters():
                if i%2 == 1:
                    parameter = torch.cat((parameter, ps), 0)
                i += 1
            
            #scheduler = StepLR(optimizers[n_worker], step_size=1, gamma=args.gamma)
            #models[n_worker].load_state_dict(global_model.state_dict())
            
            if epoch == 1: 
                Updated_grad, _ =train(args, models[n_worker], device, train_loader, optimizers[n_worker], epoch, k, freq2, k2,n_worker+1)
            else:
                Updated_grad, _ =train(args, models[n_worker], device, train_loader, optimizers[n_worker], epoch, k, freq2, k2,n_worker+1)
            if epoch == 1 and n_worker == 0:
                _, top_indeces = TopK_chooser(Updated_grad,k, freq2)
                freq3[top_indeces] = 1
            compressed_grad, topk_indeces = TopK_chooser(Updated_grad,k3, freq)
            #if epoch > 1:
            print("Train Epoch: ", epoch , "for worker ", n_worker+1, "matched: ", np.intersect1d(top_indeces, topk_indeces).size)
            #locator = top_indeces.argsort()[topk_indeces.argsort().argsort()]
            
            """
            x = np.array(range(k2))
            plt.title('Updated Gradient at epoch %d' %epoch + ', worker %d' %(n_worker+1)) 
            plt.xlabel("indices", fontsize=15) 
            plt.ylabel("updated indices", fontsize=15) 
            plt.plot(x,freq3,color = 'Green', alpha = 0.25)
            plt.plot(x,freq,color = 'red', alpha = 1)
            plt.set_rasterized =True 
            #plt.hold = False
            #plt.plot(x,freq3,color = 'Green', alpha = 0.25)
            #plt.plot(x,freq3)
            plt.legend(['preserved indices', 'k indices'], loc='upper left', fontsize=15)
            #plt.savefig('results2/%d.pdf' %j, rasterized=True, format='pdf')
            plt.show()
            """
            average_grad += (1/num_worker)*compressed_grad
            j +=1
            #test(models[n_worker], device, test_loader)
            #scheduler.step()
        #plt.show()
        
        """
        if epoch == 1:
            pruned_parameters = torch.zeros([39760])
            pruned_parameters[top_indeces] = parameter[top_indeces]
            [w1,wprime] = pruned_parameters.split(39200)
            w1 = w1.reshape([50,784])
            [w2,wprime2] = wprime.split(500)
            w2 = w2.reshape([10,50])
            [w3,w4] = wprime2.split(50)
            global_model.fc3.weight.data = w1.clone();
            global_model.fc4.weight.data = w2.clone();
            global_model.fc3.bias.data = w3.clone();
            global_model.fc4.bias.data = w4.clone();
        """    
        
        
        #update global model
        optimizer_global.zero_grad()
        [w1,wprime] = average_grad.split(39200)
        w1 = w1.reshape([50,784])
        [w2,wprime2] = wprime.split(500)
        w2 = w2.reshape([10,50])
        [w3,w4] = wprime2.split(50)
        global_model.fc3.weight.grad = w1.clone()
        global_model.fc4.weight.grad = w2.clone()
        global_model.fc3.bias.grad = w3.clone()
        global_model.fc4.bias.grad = w4.clone()
        
        
        """
        if epoch == 1:
            pruned_gradient = torch.zeros([39760])
            pruned_gradient[top_indeces] = average_grad[top_indeces]
            [w1,wprime] = pruned_parameters.split(39200)
            w1 = w1.reshape([50,784])
            [w2,wprime2] = wprime.split(500)
            w2 = w2.reshape([10,50])
            [w3,w4] = wprime2.split(50)
            global_model.fc3.weight.grad = w1.clone()
            global_model.fc4.weight.grad = w2.clone()
            global_model.fc3.bias.grad = w3.clone()
            global_model.fc4.bias.grad = w4.clone()
        
        
        
        else:
            #update global model
            optimizer_global.zero_grad()
            [w1,wprime] = average_grad.split(39200)
            w1 = w1.reshape([50,784])
            [w2,wprime2] = wprime.split(500)
            w2 = w2.reshape([10,50])
            [w3,w4] = wprime2.split(50)
            global_model.fc3.weight.grad = w1.clone()
            global_model.fc4.weight.grad = w2.clone()
            global_model.fc3.bias.grad = w3.clone()
            global_model.fc4.bias.grad = w4.clone()
       """ 

        optimizer_global.step()
        test(global_model, device, test_loader)
        #broadcast global parameters for the next epoch
        for n_worker in range(num_worker):
            #optimizers[n_worker].zero_grad()
            
            models[n_worker].load_state_dict(global_model.state_dict())
            models[n_worker].fc3.weight.grad= global_model.fc3.weight.grad.clone()
            models[n_worker].fc3.bias.grad = global_model.fc3.bias.grad.clone()
            models[n_worker].fc4.weight.grad = global_model.fc4.weight.grad.clone()
            models[n_worker].fc4.bias.grad = global_model.fc4.bias.grad.clone()
            
            #optimizers[n_worker].step()
    #x = np.array(range(39760))
    #plt.title("Gradiant update histogram") 
    #plt.xlabel("indices") 
    #plt.ylabel("frequency of gradient update") 
    #plt.plot(x,freq) 
    #plt.show()


if __name__ == '__main__':
    main()