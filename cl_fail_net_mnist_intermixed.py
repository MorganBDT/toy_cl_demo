import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import random

n_tasks = 20
n_epochs = 30
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.005
momentum = 0.5

log_interval = 10

if torch.cuda.is_available():
    device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    print("{} GPU(s) available.".format(torch.cuda.device_count()))
    print("Running on GPU: " + torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("Running on the CPU")

# Make predictable random numbers for reproducibility
torch.manual_seed(1)
np.random.seed(0)
random.seed(0)
torch.backends.cudnn.enabled = False  # for tutorial

# TODO: look into num_workers > 1 parameter option to speed up data loading
train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./data/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./data/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_test, shuffle=True)

examples = enumerate(train_loader)
batch_idx, (example_data, example_targets) = next(examples)

imgvec_size = 28*28
perms = [np.identity(imgvec_size)]  # First permutation is identity
for i in range(19):
    perm = np.identity(imgvec_size)
    np.random.shuffle(perm)
    perms.append(perm)


def permute_image(img, perm_id):
    return torch.from_numpy(np.reshape(
        np.matmul(perms[perm_id], np.ravel(img.numpy())), img.shape))


def permute_batch(batch, perm_id, intermix=False):
    for im in range(batch.shape[0]):
        if intermix:
            perm_id = random.randint(0, n_tasks-1)
        batch[im] = permute_image(batch[im], perm_id)
    return batch


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        # x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2(x))
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


network = Net().to(device)
optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

train_losses = []
train_accs = []
train_counter = []
test_losses = []
test_accs = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

def train(epoch, perm_id=0, intermix=False):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = permute_batch(data, perm_id, intermix=intermix)
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


def test(perm_id=0, intermix=False):
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = permute_batch(data, perm_id, intermix=intermix)
            data = data.to(device)
            target = target.to(device)
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    test_acc = float(100. * correct / len(test_loader.dataset))
    test_accs.append(test_acc)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss, test_acc


test(intermix=True)
tstart = time.time()
for epoch in range(1, n_epochs+1):
    train(epoch, intermix=True)
    test(intermix=True)
print("Runtime: {} seconds".format(round(time.time()-tstart, 2)))

fig = plt.figure()
#plt.plot(train_counter, train_accs, color='blue')
plt.scatter(test_counter[0:len(test_losses)], test_accs, color='red')
#plt.legend(['Training accuracy (%)', 'Testing accuracy (%)'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('Accuracy')
plt.savefig("./results/mnist_intermixed_accuracy.png")

fig = plt.figure()
plt.plot(train_counter, train_losses, color='blue')
plt.scatter(test_counter[0:len(test_losses)], test_losses, color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss')
plt.savefig("./results/mnist_intermixed_loss.png")

plt.show()
