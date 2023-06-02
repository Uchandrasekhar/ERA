                                  #CODE BLOCK:2
def utilizations():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # These 2 lines of code are added from last block
    print("CUDA Available?", use_cuda)
    #added above statement to check the status of CUDA
    return device
                                  #CODE BLOCK:3
# Train data transformations
#def transforms (train_transforms,test_transforms):
    train_transforms = transforms.Compose([
                    transforms.RandomApply([transforms.CenterCrop(22), ], p=0.1),
                    transforms.Resize((28, 28)),
                    transforms.RandomRotation((-15., 15.), fill=0),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                 ])

    # Train data transformations
    train_transforms = transforms.Compose([
                        transforms.RandomApply([transforms.CenterCrop(22), ], p=0.1),
                        transforms.Resize((28, 28)),
                        transforms.RandomRotation((-15., 15.), fill=0),
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,)),
                            ])
    
# Test data transformations
    test_transforms = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                    ])
    return(train_transforms,test_transforms)
    print(len(train_transformers,len(test_transformers)))
                                 #CODE BLOCK:4
#def data(train_data,test_data):
    train_data = datasets.MNIST('../data', train=True, download=True, transform=train_transforms)
    test_data = datasets.MNIST('../data', train=False, download=True, transform=test_transforms) 
    #return train_data,test_data
                                #CODE BLOCK:5
#def loader(test_loader,train_loader):
    batch_size = 512

    kwargs = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 2, 'pin_memory': True}

    test_loader = torch.utils.data.DataLoader(test_data, **kwargs)
    train_loader = torch.utils.data.DataLoader(train_data, **kwargs) 
    return test_loader,train_loader
                                 #CODE BLOCK:6
#def plot():

    batch_data, batch_label = next(iter(train_loader)) 

    fig = plt.figure()

    for i in range(12):
        plt.subplot(3,4,i+1)
        plt.tight_layout()
        plt.imshow(batch_data[i].squeeze(0), cmap='gray')
        plt.title(batch_label[i].item())
        plt.xticks([])
        plt.yticks([])   
    return(fig)                                                                                                              