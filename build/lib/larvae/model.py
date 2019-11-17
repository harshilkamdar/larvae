import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F

class VAE(nn.Module):
    """
    Simple variational auto-encoder. 

    Code based on https://github.com/pytorch/examples/tree/master/vae
    """
    def __init__(self, input_dim=7214, hidden_dim=400, latent_dim=20):
        super(VAE, self).__init__()

        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim) #Encoder input_dim --> hidden_dim
        self.fc21 = nn.Linear(hidden_dim, latent_dim) #mu layer
        self.fc22 = nn.Linear(hidden_dim, latent_dim) #log(sigma) layer
        self.fc3 = nn.Linear(latent_dim, hidden_dim) #Decoder latent_dim --> hidden_dim
        self.fc4 = nn.Linear(hidden_dim, input_dim) #hidden_dim --> output
        

    def encode(self, x):
        """
        encode input data into mu, sigma (from input_dim --> hidden_dim --> latent_dim)
        """
        h1 = F.relu(self.fc1(x)) #ReLU non-lin
        return self.fc21(h1), self.fc22(h1) #mu, sigma

    def reparameterize(self, mu, logvar):
        """
        https://stats.stackexchange.com/questions/16334/how-to-sample-from-a-normal-distribution-with-known-mean-and-variance-using-a-co/16338#16338
        """
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        """
        decode from latent space back to data space (from latent_dim --> hidden_dim --> input_dim)
        """
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        """
        forward pass
        """
        mu, logvar = self.encode(x.view(-1, self.input_dim)) #go to latent
        z = self.reparameterize(mu, logvar) #reparametrize
        return self.decode(z), mu, logvar #go back to data
    
    def loss_function(self, recon_x, x, mu, logvar, beta=1):
        """
        Loss = reconstruction + KLD
        """
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, self.input_dim), reduction='sum') #reconstruction loss 
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) #KL diveregence 
        return BCE + beta*KLD 
