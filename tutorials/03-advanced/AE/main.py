import torch
import torchvision
import os
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
from torchvision import transforms
from torch import nn
from torchvision.utils import save_image


num_epochs = 10
batch_size = 128
learning_rate = 1e-3
IMG_SIZE = 784
IMG_WIDTH = 28
IMG_HEIGHT = 28
print_freq =100

# MNIST data_loader
dataset = torchvision.datasets.MNIST(root='/home/roit/datasets/mnist/',
                                     train=True,
                                     transform=transforms.ToTensor(),
                                     download=True)



# Data loader
data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                          batch_size=batch_size,
										  num_workers=10,
                                          shuffle=True)



# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def to_img(x):#784 tensor to 28x28 tensor
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x

class AutoEncoder(nn.Module):
	"""
	"""
	def __init__(self, latent_num=16):
		"""
		TODO: doconvolution
		"""
		super(AutoEncoder, self).__init__()
		#encoder layers
		self.fc1 = nn.Linear(IMG_SIZE, 256)
		self.fc1.weight.data.normal_(0.0, 0.05)

		self.fc2 = nn.Linear(256, 64)
		self.fc2.weight.data.normal_(0.0, 0.05)

		#decoder layers
		self.fc3 = nn.Linear(64, 256)
		self.fc3.weight.data.normal_(0.0, 0.05)

		self.fc4 = nn.Linear(256, IMG_SIZE)
		self.fc4.weight.data.normal_(0.0, 0.05)


	def forward(self, x):
		#encoder
		h1 = F.relu(self.fc1(x))  # 784 -> 256
		h2 = F.relu(self.fc2(h1)) # 256 -> 64
		#decoder
		h3 = F.relu(self.fc3(h2)) # 64 -> 256
		h4 = F.relu(self.fc4(h3)) # 256 -> 784
		output = h4
		# output = F.sigmoid(h6)
		return output

# ref: http://kvfrans.com/variational-autoencoders-explained/
# 	1) encoder loss = mean square error from original image and decoder image
# 	2) decoder loss = KL divergence 

def loss_function(output, x):
		"""
		"""
		encoder_loss = nn.MSELoss(size_average=True)
		mse = encoder_loss(output, x)
		return mse

# way to construct DNN network
# 	1) topology 
# 	2) loss function
#		3) optimizer
# 	4) forward
# 	5) free zero grad of variable
# 	6) backward
#if (os.path.exists('ae.pth')==True):
#	model = torch.load('ae.pth')
#else:
model = AutoEncoder().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(num_epochs):
	train_loss = 0

	for batch_idx, (x,_) in enumerate(data_loader):# x.size [b,1,28,28]
		x = x.to(device).view(-1, IMG_SIZE)#to_img 的反向操作, 拉成一列一阶张量784

		output = model(x)
		# backward
		loss = loss_function(output, x)

		train_loss += loss#for avg_loss of each batch

		# free zero grad
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if batch_idx % 100 == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch,
				batch_idx ,
				len(data_loader),
				100. * batch_idx / len(data_loader),
				loss.data / len(x)))


	print('====> Epoch: {} Average loss: {:.4f}'.format(
		epoch, train_loss / len(data_loader)))
	#if epoch %  == 0:
	# for the last batch, which only have 96 imgs instead of 128: 60000- 128*(len(data_loader)-1) = 96
	save = to_img(output.cpu().data)
	save_image(save, 'ae_img/output_image_{}.png'.format(epoch))
	save = to_img(x.cpu().data)
	save_image(save, 'vae_img/x_image_{}.png'.format(epoch))

# save model
torch.save(model.state_dict(), './ae.pth')

