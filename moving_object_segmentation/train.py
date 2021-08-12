import torch
from torch.utils.data import DataLoader
from dataloader import synDataset
from loss import calc_loss,dice_loss

def label_conv(labels,targetimage, mask1=True):
    if mask1:
        # background        
        labels[:,0,:,:] = (targetimage > 0)==0
        labels[:,0,:,:] = (targetimage == 0)==1  
        # mask1
        labels[:,1,:,:] = (targetimage > 0)==1
        labels[:,1,:,:] = (targetimage == 0)==0
    else:
        # background        
        labels[:,0,:,:] = (targetimage < 1)==1
        labels[:,0,:,:] = (targetimage == 1)==0  
        # mask1
        labels[:,1,:,:] = (targetimage < 1)==0
        labels[:,1,:,:] = (targetimage == 1)==1
    return labels

def train(model):
    batch_size = 1
    dataloaders = {
        'train': DataLoader(synDataset(inp_dim=(448,448)), batch_size=batch_size, shuffle=True, num_workers=2),
    }
    gpu_available = torch.cuda.is_available()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    num_epochs = 15
    loss_values =[]
    running_loss = 0.0

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for param_group in optimizer.param_groups:
            print("LR", param_group['lr'])
        scheduler.step()
        for i, data in enumerate(dataloaders['train']):
            inputimage, flowimage, mask1, mask2 = data
            
            if gpu_available:
                inputimage = inputimage.cuda()
                flowimage = flowimage.cuda()
                mask1 = mask1.cuda()
                mask2 = mask2.cuda()

            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputimage,flowimage)
            
            labels = torch.empty_like(outputs[0],dtype=torch.float)
            lab_channel1 = (label_conv(labels,mask1,True))
            labels = torch.empty_like(outputs[1],dtype=torch.float)
            lab_channel2 = (label_conv(labels,mask2,True))

            if gpu_available:
                lab_channel1 = lab_channel1.cuda()
                lab_channel2 = lab_channel2.cuda()


            #loss_dec1 = dice_loss(torch.sigmoid(outputs[0]),lab_channel1)
            #loss_dec2 = dice_loss(torch.sigmoid(outputs[1]),lab_channel2)
            loss_dec1 = calc_loss(outputs[0],lab_channel1)
            loss_dec2 = dice_loss(outputs[1],lab_channel2)
            losses = loss_dec1+loss_dec2

            #loss += lmbd * reg_loss
            losses.backward()
            optimizer.step()

            # print statistics
            running_loss += losses.item()
            loss_values.append(losses.item())
            if i % 10 == 9:
                print('[%d, %5d] loss: %.10f' %(epoch + 1, i + 1, running_loss / 10))
                #loss_values.append(running_loss / co)
                running_loss = 0.0
        epoch_loss = running_loss / len(dataloaders['train'])
        print('epoch loss: %.4f'%(epoch_loss))