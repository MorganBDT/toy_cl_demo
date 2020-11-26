import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import sys

if len(sys.argv) > 1:
    l2 = float(sys.argv[1])
    print("l2 value: {}".format(l2))
else:
    l2 = 0

n_epochs = 15
batch_size_train = 64
batch_size_test = 250
learning_rate = 0.05
momentum = 0.5
#l2 = 0.001
log_interval = 10 #? from tutorial

if torch.cuda.is_available():
    device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    print("{} GPU(s) available.".format(torch.cuda.device_count()))
    print("Running on GPU: " + torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("Running on the CPU")

torch.manual_seed(1) # for tutorial
torch.backends.cudnn.enabled = False # for tutorial

# TODO: look into num_workers > 1 parameter option to speed up data loading
train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.CIFAR10('./data/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                             ])),
  batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.CIFAR10('./data/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                             ])),
  batch_size=batch_size_test, shuffle=True)

examples = enumerate(train_loader)
batch_idx, (example_data, example_targets) = next(examples)

#print(batch_idx)
#print(example_data.shape)
#print(example_targets.shape)

# Show first six examples in training set
# fig = plt.figure()
# for i in range(6):
#     plt.subplot(2, 3, i+1)
#     plt.tight_layout()
#     plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
#     plt.title(int(example_targets[i]))
#     plt.xticks([])
#     plt.yticks([])
# plt.show()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.conv1_drop = nn.Dropout2d()
        self.conv1_bn = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.conv2_bn = nn.BatchNorm2d(16)

        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3_drop = nn.Dropout2d()
        self.conv3_bn = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4_bn = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv5_drop = nn.Dropout2d()
        self.conv5_bn = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv6_bn = nn.BatchNorm2d(256)

        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv7_drop = nn.Dropout2d()
        self.conv7_bn = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.conv8_bn = nn.BatchNorm2d(1024)

        self.fc1 = nn.Linear(4096, 128)
        self.fc2 = nn.Linear(128, 10)
        #self.fc1_drop = nn.Dropout()

    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)))
        #x = self.conv1_drop(x)
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv3_bn(self.conv3(x)))
        #x = self.conv3_drop(x)
        x = F.relu(self.conv4_bn(self.conv4(x)))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv5_bn(self.conv5(x)))
        #x = self.conv5_drop(x)
        x = F.relu(self.conv6_bn(self.conv6(x)))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv7_bn(self.conv7(x)))
        #x = self.conv5_drop(x)
        x = F.relu(self.conv8_bn(self.conv8(x)))
        x = F.max_pool2d(x, 2)

        x = x.view(-1, 4096)
        x = F.relu(self.fc1(x))
        #x = self.fc1_drop(x)
        x = F.log_softmax(self.fc2(x))
        return x


network = Net().to(device)
optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum, weight_decay=l2)

if len(sys.argv) > 2 and sys.argv[2] == "--continue":
    network_state_dict = torch.load("model.pth")
    network.load_state_dict(network_state_dict)

    optimizer_state_dict = torch.load("optimizer.pth")
    optimizer.load_state_dict(optimizer_state_dict)

train_losses = []
train_accs = []
train_counter = []
test_losses = []
test_accs = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]


def train(epoch):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = network(data)  # forward pass
        loss = F.nll_loss(output, target)  # negative log-likelihood loss
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_accs.append(100. * batch_idx / len(train_loader))
            train_counter.append(
                (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))
            torch.save(network.state_dict(), './results/model.pth')
            torch.save(optimizer.state_dict(), './results/optimizer.pth')


def test():
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    test_accs.append(float(100. * correct / len(test_loader.dataset)))
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return correct / len(test_loader.dataset)

tstart = time.time()
test()
prev_acc = 0
for epoch in range(1, n_epochs+1):
    train(epoch)
    acc = test()
    # if abs(acc - prev_acc) < 0.001:
    #     print("Accuracy changed by less than 0.1% between epochs. Exiting...")
    #     break
    # prev_acc = acc

print("Runtime: {} seconds".format(round(time.time()-tstart, 2)))

fig = plt.figure()
#plt.plot(train_counter, train_accs, color='blue')
plt.scatter(test_counter[0:len(test_losses)], test_accs, color='red')
#plt.legend(['Training accuracy (%)', 'Testing accuracy (%)'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('Accuracy')
#plt.show()
plt.savefig("./results/cifar_dropout_acc.png")

fig = plt.figure()
plt.plot(train_counter, train_losses, color='blue')
plt.scatter(test_counter[0:len(test_losses)], test_losses, color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss')
#plt.show()
plt.savefig("./results/cifar_dropout_loss.png")
