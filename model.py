from torchcfm.models.models import MLP
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from vit_pytorch import ViT
# Define the Transformer Encoder
class TransformerEncoder(nn.Module):
    def __init__(self, image_size=128, patch_size=16, num_classes=256, dim=512, depth=6, heads=8, mlp_dim=1024, dropout=0.1, emb_dropout=0.1):
        super(TransformerEncoder, self).__init__()
        
        # Define the Vision Transformer
        self.transformer = ViT(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=dim,  # Output the embeddings of dim size
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            emb_dropout=emb_dropout,
            channels=2  # Set to 2 channels for your specific input
        )
        
        # Define a linear layer to map the features to the desired output dimension
        self.fc = nn.Linear(dim, num_classes)
    
    def forward(self, x):
        batch_size, channels, frames, height, width = x.size()
        
        # Reshape input to (batch_size * frames, channels, height, width)
        x = x.view(batch_size * frames, channels, height, width)
        
        # Extract embeddings for each frame using the transformer
        embeddings = self.transformer(x)
        
        # Reshape embeddings back to (batch_size, frames, embedding_dim)
        embeddings = embeddings.view(batch_size, frames, -1)
        
        # Pool the embeddings across the time dimension (frames)
        pooled_embeddings = torch.mean(embeddings, dim=1)
        
        # Map the pooled embeddings to the output dimension
        output = self.fc(pooled_embeddings)
        print('encoder output',output.shape)
        return output


# Define the combined model
class CombinedModel(nn.Module):
    def __init__(self, features=256,outputmlp=300,number_of_frames=100):
        super(CombinedModel, self).__init__()
        self.feature = features  # Output dimension of encoder
        self.outputmlp = outputmlp
        self.number_of_frames = number_of_frames
        self.encoder =  TransformerEncoder(num_classes=self.feature)  # Output dimension of encoder
        self.input_mlp = self.feature + self.outputmlp + self.number_of_frames  # t is 100 length if 100 frames
        print('input_mlp',self.input_mlp)
        print('outputmlp',self.outputmlp)
        self.mlp = MLP(dim=self.input_mlp ,out_dim=self.outputmlp) 

    def forward(self, x, t_expanded, noisy_theta_noise=None, noisy_theta_x_noise=None, noisy_theta_y_noise=None):
        embeddings = self.encoder(x)
        
        inputs = [embeddings, t_expanded]
        
        if noisy_theta_noise is not None:
            inputs.append(noisy_theta_noise)
        if noisy_theta_x_noise is not None:
            inputs.append(noisy_theta_x_noise)
        if noisy_theta_y_noise is not None:
            inputs.append(noisy_theta_y_noise)
            
        concatenated_input = torch.cat(inputs, dim=-1)
        print('real_input_mlp',concatenated_input.shape)
        output = self.mlp(concatenated_input)
        
        return output

