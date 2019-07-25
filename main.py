from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import codecs


class Net(nn.Module):
    def __init__(self, hidden_layer_dim):
        super(Net, self).__init__()
        # self.conv1 = nn.Conv2d(1, 20, 5, 1)
        # self.conv2 = nn.Conv2d(20, 50, 5, 1)
        # self.fc1 = nn.Linear(4*4*50, 500)
        # self.fc2 = nn.Linear(500, 10)

        self.input_layer = nn.Linear(784, hidden_layer_dim, bias=True)
        nn.init.kaiming_normal_(self.input_layer.weight.data)
        self.hidden_layer = nn.Linear(hidden_layer_dim, 10, bias=True)
        nn.init.kaiming_normal_(self.hidden_layer.weight.data)

    def forward(self, x):
        # x = F.relu(self.conv1(x))
        # x = F.max_pool2d(x, 2, 2)
        # x = F.relu(self.conv2(x))
        # x = F.max_pool2d(x, 2, 2)
        # x = x.view(-1, 4*4*50)
        # x = F.relu(self.fc1(x))
        # x = self.fc2(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.input_layer(x))
        # x = F.tanh(self.input_layer(x))
        x = self.hidden_layer(x)
        # return x

        return F.log_softmax(x, dim=1)


def adjust_learning_rate(optimizer, epoch, args):
        """decrease the learning rate at specific epochs"""
        lr = args.lr

        target_lr_phase1 = lr / 10
        target_lr_phase2 = target_lr_phase1 / 10
        gamma_phase1 = (target_lr_phase1 / args.lr) ** (1 / 101)
        gamma_phase2 = (target_lr_phase2 / target_lr_phase1) ** (1 / 50)
        if epoch <= 100:
            lr = lr * gamma_phase1 ** (epoch + 1)
        elif epoch <= 150:
            lr = target_lr_phase1 * gamma_phase2 ** (epoch - 100)
        else:
            lr = target_lr_phase2

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    
def train(args, model, device, train_loader, optimizer, epoch, criterion):
    model.train()
    train_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        # loss = criterion(output, target)
        train_loss += (loss.item() * data.shape[0])
        loss.backward()
        optimizer.step()
        pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        # if batch_idx % args.log_interval == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader), loss.item()))

    train_loss /=len(train_loader.dataset)
    acc = 100. * correct / len(train_loader.dataset)
    return train_loss, acc


def test(args, model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            # test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)

    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     test_loss, correct, len(test_loader.dataset),
    #     100. * correct / len(test_loader.dataset)))

    return test_loss, acc


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    # parser.add_argument('--seed', type=int, default=1, metavar='S',
    #                     help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument("--hidden_layer_dim", default=20, type=int, 
                        help="hidden_layer_dim")
    
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument("--num_experiment", default=1, type=int, 
                        help="num_experiment")
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    SEED = {0:1, 1:12, 2:123, 3:1234, 4:4321}
    torch.manual_seed(SEED[args.num_experiment])

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)


    model = Net(args.hidden_layer_dim).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    criterion = nn.CrossEntropyLoss(reduction='sum')

    with codecs.open(str(args.num_experiment) + '_' + str(args.hidden_layer_dim) + '_adjust.log', 'w', 'utf-8') as wf_log:
        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train(args, model, device, train_loader, optimizer, epoch, criterion)
            test_loss, test_acc = test(args, model, device, test_loader, criterion)
            # adjust_learning_rate(optimizer, epoch-1, args)
            print(epoch, '==>', train_loss, train_acc, test_loss, test_acc)
            wf_log.write(','.join([str(train_loss), str(train_acc), str(test_loss), str(test_acc)]) + '\n')

    if (args.save_model):
        torch.save(model.state_dict(),"mnist_cnn.pt")
        
if __name__ == '__main__':
    main()
