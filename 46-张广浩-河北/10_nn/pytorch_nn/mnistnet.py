import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


def dataset():
	transform = transforms.Compose(
		[
			transforms.ToTensor(),
			transforms.Normalize([0, ], [1, ])
		]
	)
	trainset = torchvision.datasets.MNIST(root='pytorch_nn/data', train=True, download=True, transform=transform)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)
	testset = torchvision.datasets.MNIST(root='pytorch_nn/data', train=False, download=True, transform=transform)
	testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True, num_workers=2)
	return trainloader, testloader


class Model:
	def __init__(self, net, loss_function, optimist):
		self.net = net
		self.cost = self.create_loss_function(loss_function)
		self.optimizer = self.create_optimizer(optimist)

	def create_loss_function(self, cost):
		support_cost = {
			'CROSS_ENTROPY': nn.CrossEntropyLoss(),
			'MSE': nn.MSELoss()
		}
		return support_cost[cost].cuda()

	def create_optimizer(self, optimist, **rests):
		support_optim = {
			'SGD': optim.SGD(self.net.parameters(), lr=0.12, **rests),
			'ADAM': optim.Adam(self.net.parameters(), lr=0.01, **rests),
			'RMSP': optim.RMSprop(self.net.parameters(), lr=0.001, **rests)
		}
		return support_optim[optimist]

	def train(self, train_loader, epoches=3):
		for epoch in range(epoches):
			running_loss = 0.0
			for i, data in enumerate(train_loader, 0):
				inputs, labels = data
				inputs = inputs.cuda()
				labels = labels.cuda()
				self.optimizer.zero_grad()
				outputs = self.net(inputs)
				loss = self.cost(outputs, labels)
				loss.backward()
				self.optimizer.step()

				running_loss += loss.item()
				if i % 100 == 0:
					print('[epoch %d, %.2f%%] loss: %.3f' %
					      (epoch + 1, (i + 1) * 1. / len(train_loader), running_loss / 100))
					running_loss = 0.0

		print('Finished Training')

	def evaluate(self, test_loader):
		print('Evaluating ...')
		correct = 0
		total = 0
		with torch.no_grad():  # no grad when test and predict
			for data in test_loader:
				images, labels = data
				images = images.cuda()
				labels = labels.cuda()
				outputs = self.net(images)
				predicted = torch.argmax(outputs, 1)
				total += labels.size(0)
				correct += (predicted == labels).sum().item()

		print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))


class MnistNet(torch.nn.Module):
	def __init__(self):
		super(MnistNet, self).__init__()
		self.fc1 = torch.nn.Linear(28 * 28, 512)
		self.fc2 = torch.nn.Linear(512, 512)
		# self.fc3 = torch.nn.Linear(512, 512)
		self.fc3 = torch.nn.Linear(512, 10)

	def forward(self, x):
		x = x.view(-1, 28 * 28)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		# x = F.relu(self.fc3(x))
		x = F.softmax(self.fc3(x), dim=1)
		return x


if __name__ == '__main__':
	# train for mnist
	net = MnistNet()
	net.cuda()
	model = Model(net, 'CROSS_ENTROPY', 'SGD')
	train_loader, test_loader = dataset()
	model.train(train_loader)
	model.evaluate(test_loader)
