import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import time

n_epochs = 2
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10 #? from tutorial

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
torch.backends.cudnn.enabled = False  # from tutorial

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

imgvec_size = 28*28
perms = [np.identity(imgvec_size)]  # First permutation is identity
for i in range(19):
    perm = np.identity(imgvec_size)
    np.random.shuffle(perm)
    perms.append(perm)


def permute_image(img, perm_id):
    return torch.from_numpy(np.reshape(
        np.matmul(perms[perm_id], np.ravel(img.numpy())), img.shape))


def permute_batch(batch, perm_id):
    for im in range(batch.shape[0]):
        batch[im] = permute_image(batch[im], perm_id)
    return batch


examples = enumerate(train_loader)
batch_idx, (example_data, example_targets) = next(examples)
print(example_data.shape)

# Show several permutations of image
plt.figure()
plt.subplot(2, 3, 1)
plt.imshow(permute_image(example_data[0][0], 0), cmap='gray', interpolation='none')
plt.subplot(2, 3, 2)
plt.imshow(permute_image(example_data[0][0], 1), cmap='gray', interpolation='none')
plt.subplot(2, 3, 3)
plt.imshow(permute_image(example_data[0][0], 2), cmap='gray', interpolation='none')

plt.subplot(2, 3, 4)
plt.imshow(permute_batch(example_data.clone(), 0)[0][0], cmap='gray', interpolation='none')
plt.subplot(2, 3, 5)
plt.imshow(permute_batch(example_data.clone(), 1)[0][0], cmap='gray', interpolation='none')
plt.subplot(2, 3, 6)
plt.imshow(permute_batch(example_data.clone(), 2)[0][0], cmap='gray', interpolation='none')
plt.show()

# Show first six examples in training set
fig = plt.figure()
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.tight_layout()
    plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
    plt.title(int(example_targets[i]))
    plt.xticks([])
    plt.yticks([])
plt.show()