"""
Author  :    Ziping Xu
Email   :    zipingxu@umich.edu
Date    :    Jan 17, 2020
Record  :    Deepfake: test module
"""

from Utilize.Prediction import *
from Utilize.Data import *
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

device = 'cuda'

def train_transfer_model(model, dataset_train, dataset_test, 
                          optimizer, criterion, post_function,
                          n_epochs=10, batch_size=10, 
                          stop_criterion = 0.001, checkpoint = None,
                          post_every_iterations = 80, device = device):

    dataloader = DataLoader(dataset_train, batch_size, shuffle=True, num_workers = 8)
    loss_list = [1]
    
    for epoch in range(n_epochs):
        
        model = model.train()
        running_loss = 0.0
        
        for i, batch in enumerate(dataloader):
            
            inputs = batch['image']
            labels = batch['label']
            
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
        
            outputs = model(inputs)
            outputs = post_function(outputs)[:,1]
        
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
        
            if i % post_every_iterations == post_every_iterations-1:    # print every 50 mini-batches
                print('[Epoch: %d, minibatch: %6d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / post_every_iterations))
                running_loss = 0

        loss, pred_label, true_label = validate_on_test_set(model, dataset_test, device = device, ratio = 0.1)
        print('[Epoch: %d] test on val_set loss: %.3f' %(epoch + 1, loss))
        
        if loss_list[epoch]-loss < 0.001:
          break
            
        loss_list.append(loss)

        if type(checkpoint) is not type(None):

            checkpoint['epoch_num'] = epoch
            checkpoint['state_dict'] = model.state_dict()
            checkpoint['loss_list'] = checkpoint['loss_list']+[loss]

def validate_on_test_set(model, dataset_test, device = device, ratio = 1, post_function = nn.Softmax(dim=1)):
    
    model = model.eval()
    
    pred_label = []
    true_label = []
    
    dataloader = DataLoader(dataset_test, batch_size=1, shuffle=True, num_workers = 8)
    
    for i, sample in enumerate(dataloader):
        
        image = sample['image']
        label = sample['label'].item()
        image = image.to(device)
        
        output = model(image)
        output = post_function(output)[:,0]
        
        prediction = float(output.detach().cpu())
        pred_label.append(prediction)
        true_label.append(label)
        
        if i == int(len(dataset_test)*ratio):
            break
        
    #dataset_test.image_df['pred_label'] = pred_label
    
    return log_loss(true_label, [1-x for x in pred_label]), pred_label, true_label
