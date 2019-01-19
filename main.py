from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


class LeNet(nn.Module):
	def __init__(self, args):
		super(LeNet, self).__init__()

		self.p = args.dropout
		self.batch_norm = args.batch_norm

		self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
		self.conv1_bn = nn.BatchNorm2d(6)
		self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
		self.conv2_bn = nn.BatchNorm2d(16)
		self.conv2_drop = nn.Dropout2d(p=self.p)
		self.fc1 = nn.Linear(16*5*5, 120)
		self.fc1_bn = nn.BatchNorm1d(120)
		self.fc2 = nn.Linear(120, 84)
		self.fc2_bn = nn.BatchNorm1d(84)
		self.fc3 = nn.Linear(84, 10)

	def forward(self, x):
		x = F.relu(F.max_pool2d(self.conv1(x), 2))
		if self.batch_norm:
			x = self.conv1_bn(x)
		x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
		if self.batch_norm:
			x = self.conv2_bn(x)
		x = x.view(-1, 16*5*5)
		x = F.relu(self.fc1(x))
		if self.batch_norm:
			x = self.fc1_bn(x)
		x = F.dropout(x, training=self.training, p=self.p)
		x = F.relu(self.fc2(x))
		if self.batch_norm:
			x = self.fc2_bn(x)
		x = F.dropout(x, training=self.training, p=self.p)
		x = self.fc3(x)
		return F.log_softmax(x)


def train(args, model, device, train_loader, optimizer, epoch):
	model.train()
	correct = 0
	for batch_idx, (data, target) in enumerate(train_loader):
		data, target = data.to(device), target.to(device)
		optimizer.zero_grad()
		output = model(data)

		pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
		correct += pred.eq(target.view_as(pred)).sum().item()

		loss = F.nll_loss(output, target)
		loss.backward()
		optimizer.step()
		if batch_idx % args.log_interval == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch, batch_idx * len(data), len(train_loader.dataset),
				100. * batch_idx / len(train_loader), loss.item()))

	train_acc = correct / len(train_loader.dataset)
	return train_acc


def test(args, model, device, test_loader):
	model.eval()
	test_loss = 0
	correct = 0
	with torch.no_grad():
		for data, target in test_loader:
			data, target = data.to(device), target.to(device)
			output = model(data)
			test_loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
			pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
			correct += pred.eq(target.view_as(pred)).sum().item()

	test_loss /= len(test_loader.dataset)
	test_acc = correct / len(test_loader.dataset)
	print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
		test_loss, correct, len(test_loader.dataset),
		100. * test_acc))

	return test_acc


def main():
	# Training settings
	parser = argparse.ArgumentParser(description='PyTorch FashionMNIST')
	parser.add_argument('--batch-size', type=int, default=64, metavar='N',
						help='input batch size for training (default: 64)')
	parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
						help='input batch size for testing (default: 1000)')
	parser.add_argument('--epochs', type=int, default=10, metavar='N',
						help='number of epochs to train (default: 10)')
	parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
						help='learning rate (default: 0.01)')
	parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
						help='SGD momentum (default: 0.5)')
	parser.add_argument('--no-cuda', action='store_true', default=False,
						help='disables CUDA training')
	parser.add_argument('--seed', type=int, default=1, metavar='S',
						help='random seed (default: 1)')
	parser.add_argument('--log-interval', type=int, default=10, metavar='N',
						help='how many batches to wait before logging training status')
	parser.add_argument('--dropout', type=float, default=0, metavar='N',
						help='dropout rate')
	parser.add_argument('--weight-decay', type=float, default=0, metavar='N',
						help='weight decay')
	parser.add_argument('--batch-norm', type=bool, default=False, metavar='N',
						help='batch normalization mode')
	parser.add_argument('--saved-weights', type=str, default=None, metavar='P',
						help='path to saved weights')
	args = parser.parse_args()
	use_cuda = not args.no_cuda and torch.cuda.is_available()

	torch.manual_seed(args.seed)

	device = torch.device("cuda" if use_cuda else "cpu")

	kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
	train_loader = torch.utils.data.DataLoader(
		datasets.MNIST('data', train=True, download=True, transform=transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.2860,), (0.3205,))])),
		batch_size=args.batch_size, shuffle=True, **kwargs)

	# Calculate Mean & STD
	calculate_stats = False
	if calculate_stats:
		mean = 0.
		std = 0.
		nb_samples = 0.
		for data in train_loader:
			data = data[0]
			batch_samples = data.size(0)
			data = data.view(batch_samples, data.size(1), -1)
			mean += data.mean(2).sum(0)
			std += data.std(2).sum(0)
			nb_samples += batch_samples

		mean /= nb_samples
		std /= nb_samples

	test_loader = torch.utils.data.DataLoader(
		datasets.FashionMNIST('data', train=False, transform=transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.2860,), (0.3205,))])),
		batch_size=args.test_batch_size, shuffle=True, **kwargs)

	model = LeNet(args).to(device)
	#optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
	optimizer = optim.Adam(model.parameters(), weight_decay=args.weight_decay)

	if args.saved_weights is not None:
		model.load_state_dict(torch.load(args.saved_weights))

	train_acc = []
	test_acc = []
	for epoch in range(1, args.epochs + 1):
		train_acc.append(train(args, model, device, train_loader, optimizer, epoch))
		test_acc.append(test(args, model, device, test_loader))

	torch.save(model.state_dict(), "weights.pt")

	plt.figure()
	plt.title("Dropout rate: {} Weight Decay: {} Batch Norm: {}".format(args.dropout, args.weight_decay, args.batch_norm))
	plt.xlabel("Epoch")
	plt.ylabel("Accuracy")
	plt.plot(train_acc, label="Train")
	plt.plot(test_acc, label="Test")
	plt.grid()
	plt.legend(loc=3, prop={'size': 6})
	plt.savefig('dropout={}decay={}batch_norm={}.png'.format(args.dropout, args.weight_decay, args.batch_norm))
	plt.show()


if __name__ == '__main__':
	main()
