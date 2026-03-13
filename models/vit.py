"""Vision Transformer (ViT) for CIFAR-10 classification."""

import torch
import torch.nn as nn

from models.layers import TransformerEncoder

class ViT(nn.Module):
    def __init__(self, in_c:int=3, num_classes:int=10, img_size:int=32, patch:int=8, dropout:float=0., num_layers:int=7, hidden:int=384, mlp_hidden:int=384*4, head:int=8, is_cls_token:bool=True):
        super(ViT, self).__init__()
        self.patch = patch
        self.is_cls_token = is_cls_token
        self.patch_size = img_size // self.patch
        f = (img_size // self.patch) ** 2 * 3
        num_tokens = (self.patch ** 2) + 1 if self.is_cls_token else (self.patch ** 2)

        self.emb = nn.Linear(f, hidden)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden)) if is_cls_token else None
        self.pos_emb = nn.Parameter(torch.randn(1, num_tokens, hidden))
        enc_list = [TransformerEncoder(hidden, mlp_hidden=mlp_hidden, dropout=dropout, head=head) for _ in range(num_layers)]
        self.enc = nn.Sequential(*enc_list)
        self.fc = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, num_classes)
        )

        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)
        self.mean = torch.tensor(mean).view(in_c, 1, 1).cuda()
        self.std = torch.tensor(std).view(in_c, 1, 1).cuda()

    def forward(self, x):
        x = self.normalization(x)
        out = self.embedding(x)
        out = self.enc(out)
        out = self.classifier(out)
        return out
    
    def normalization(self, x):
        x = (x - self.mean) / self.std
        return x
    
    def embedding(self, x):
        x = self._to_words(x)  
        x = self.emb(x)  
        if self.is_cls_token:
            x = torch.cat([self.cls_token.repeat(x.size(0), 1, 1), x], dim=1) 
        x = x + self.pos_emb  
        return x
    
    def classifier(self, x):
        if self.is_cls_token:
            x = x[:, 0] 
        else:
            x = x.mean(dim=1)  
        x = self.fc(x)  
        return x

    def _to_words(self, x):
        """
        (b, c, h, w) -> (b, n, f)
        """
        out = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size).permute(0,2,3,4,5,1)
        out = out.reshape(x.size(0), self.patch**2 ,-1)
        return out

class ViT_feat(nn.Module):
    def __init__(self, in_c:int=3, num_classes:int=10, img_size:int=32, patch:int=8, dropout:float=0., num_layers:int=7, hidden:int=384, mlp_hidden:int=384*4, head:int=8, is_cls_token:bool=True):
        super(ViT_feat, self).__init__()
        self.patch = patch
        self.is_cls_token = is_cls_token
        self.patch_size = img_size // self.patch
        f = (img_size // self.patch) ** 2 * 3
        num_tokens = (self.patch ** 2) + 1 if self.is_cls_token else (self.patch ** 2)

        self.emb = nn.Linear(f, hidden)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden)) if is_cls_token else None
        self.pos_emb = nn.Parameter(torch.randn(1, num_tokens, hidden))
        enc_list = [TransformerEncoder(hidden, mlp_hidden=mlp_hidden, dropout=dropout, head=head) for _ in range(num_layers)]
        self.enc = nn.Sequential(*enc_list)
        self.fc = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, num_classes)
        )

        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)
        self.mean = torch.tensor(mean).view(in_c, 1, 1).cuda()
        self.std = torch.tensor(std).view(in_c, 1, 1).cuda()

    def forward(self, x):
        x = self.normalization(x)
        out = self.embedding(x)
        out = self.enc(out)

        return out
    
    def normalization(self, x):
        x = (x - self.mean) / self.std
        return x
    
    def embedding(self, x):
        x = self._to_words(x)  
        x = self.emb(x)  
        if self.is_cls_token:
            x = torch.cat([self.cls_token.repeat(x.size(0), 1, 1), x], dim=1) 
        x = x + self.pos_emb  
        return x

    def _to_words(self, x):
        """
        (b, c, h, w) -> (b, n, f)
        """
        out = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size).permute(0,2,3,4,5,1)
        out = out.reshape(x.size(0), self.patch**2 ,-1)
        return out
