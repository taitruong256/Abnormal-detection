import torch
import torch.nn as nn
import torch.nn.functional as F

class Sampling(nn.Module):
    def forward(self, mean, log_var, variance):
        variance_tensor = torch.tensor(variance, device=mean.device)
        epsilon = torch.normal(mean=torch.zeros_like(mean), std=torch.sqrt(variance_tensor))
        return mean + torch.exp(0.5 * log_var) * epsilon
    
class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.bn2(x)

        return x
    
class encoder_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(encoder_block, self).__init__() 
        self.conv_block = conv_block(in_channels, out_channels)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        x = self.conv_block(x)
        y = self.max_pool(x)
        return x, y 
    
class decoder_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(decoder_block, self).__init__()
        self.conv_trans = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_block = conv_block(in_channels, out_channels)
    
    def forward(self, skip_features, x):
        x = self.conv_trans(x)
        x = torch.cat((skip_features, x), dim=1)
        x = self.conv_block(x)
        return x 
    
class VAE_Unet(nn.Module):
    def __init__(self, input_shape, variance, latent_dim):
        self.input_shape = input_shape
        self.variance = variance 
        super(VAE_Unet, self).__init__()
        
        self.encoder_block1 = encoder_block(1, 64)
        self.encoder_block2 = encoder_block(64, 128)
        self.encoder_block3 = encoder_block(128, 256)
        self.encoder_block4 = encoder_block(256, 512)
        
        self.bottleneck = conv_block(512, 1024)
        
        self.fc_mean = nn.Linear(1024 * (self.input_shape // 16) * (self.input_shape // 16), latent_dim)
        self.fc_log_var = nn.Linear(1024 * (self.input_shape // 16) * (self.input_shape // 16), latent_dim)
        
        self.sampling = Sampling()
        self.fc_z = nn.Linear(latent_dim, 1024 * (self.input_shape // 16) * (self.input_shape // 16))
        
        self.decoder_block4 = decoder_block(1024, 512)
        self.decoder_block3 = decoder_block(512, 256)
        self.decoder_block2 = decoder_block(256, 128)
        self.decoder_block1 = decoder_block(128, 64)
        
        self.conv1 = nn.Conv2d(64, 1, kernel_size=1, stride=1)
        self.tanh = nn.Tanh() 

    def encode(self, x):
        x1, y1 = self.encoder_block1(x)
        x2, y2 = self.encoder_block2(y1)
        x3, y3 = self.encoder_block3(y2)
        x4, y4 = self.encoder_block4(y3)
        b1 = self.bottleneck(y4)
        b1_flat = b1.view(b1.size(0), -1)
        mean = self.fc_mean(b1_flat)
        log_var = self.fc_log_var(b1_flat)
        return mean, log_var, x1, x2, x3, x4

    def decode(self, z, x1, x2, x3, x4):
        z1 = self.fc_z(z)
        z1 = z1.view(z.size(0), 1024, self.input_shape // 16, self.input_shape // 16) 
        d4 = self.decoder_block4(x4, z1)
        d3 = self.decoder_block3(x3, d4)
        d2 = self.decoder_block2(x2, d3)
        d1 = self.decoder_block1(x1, d2)
        return torch.sigmoid(self.conv1(d1))

    def forward(self, x):
        mean, log_var, x1, x2, x3, x4 = self.encode(x)
        z = self.tanh(self.sampling(mean, log_var, self.variance))  
        d1 = self.decode(z, x1, x2, x3, x4)
        return mean, log_var, z, d1