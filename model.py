                                 #CODE BLOCK:7

def cnn_model():
  class Net(nn.Module):
    #This defines the structure of the NN.
      def __init__(self):
          super(Net, self).__init__()
          self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
          self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
          self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
          self.conv4 = nn.Conv2d(128, 256, kernel_size=3)
          self.fc1 = nn.Linear(4096, 50)
          self.fc2 = nn.Linear(50, 10)

      def forward(self, x):
          x = F.relu(self.conv1(x), 2)
          x = F.relu(F.max_pool2d(self.conv2(x), 2)) 
          x = F.relu(self.conv3(x), 2)
          x = F.relu(F.max_pool2d(self.conv4(x), 2)) 
          x = x.view(-1, 4096)
          x = F.relu(self.fc1(x))
          x = self.fc2(x)
          return F.log_softmax(x, dim=1)
                                
  model = Net().to(device)
  summary(model, input_size=(1, 28, 28))

                                
                                #CODE BLOCK:8
# Data to plot accuracy and loss graphs
def loss():
  train_losses = []
  test_losses = []
  train_acc = []
  test_acc = []

  test_incorrect_pred = {'images': [], 'ground_truths': [], 'predicted_vals': []}
                               
                               #CODE BLOCK:9
                                   


def GetCorrectPredCount(pPrediction, pLabels):
  return pPrediction.argmax(dim=1).eq(pLabels).sum().item()

def train(model, device, train_loader, optimizer, criterion):
  model.train()
  pbar = tqdm(train_loader)

  train_loss = 0
  correct = 0
  processed = 0

  for batch_idx, (data, target) in enumerate(pbar):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()

    # Predict
    pred = model(data)

    # Calculate loss
    loss = criterion(pred, target)
    train_loss+=loss.item()

    # Backpropagation
    loss.backward()
    optimizer.step()
    
    correct += GetCorrectPredCount(pred, target)
    processed += len(data)

    pbar.set_description(desc= f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

  train_acc.append(100*correct/processed)
  train_losses.append(train_loss/len(train_loader))

def test(model, device, test_loader, criterion):
    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            #removed reduction=sum()

            correct += GetCorrectPredCount(output, target)


    test_loss /= len(test_loader.dataset)
    test_acc.append(100. * correct / len(test_loader.dataset))
    test_losses.append(test_loss)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
                               #CODE BLOCK:10                                    

def optimi():
  model = Net().to(device)
  optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
  scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1, verbose=True)
  # New Line
  criterion = nn.CrossEntropyLoss()
  num_epochs = 20

  for epoch in range(1, num_epochs+1):
    print(f'Epoch {epoch}')
    train(model, device, train_loader, optimizer, criterion)
    test(model, device, test_loader, criterion)
    scheduler.step()

                               #CODE BLOCK:11
def acc_loss_plot():
  fig, axs = plt.subplots(2,2,figsize=(15,10))
  axs[0, 0].plot(train_losses)
  axs[0, 0].set_title("Training Loss")
  axs[1, 0].plot(train_acc)
  axs[1, 0].set_title("Training Accuracy")
  axs[0, 1].plot(test_losses)
  axs[0, 1].set_title("Test Loss")
  axs[1, 1].plot(test_acc)
  axs[1, 1].set_title("Test Accuracy")                            