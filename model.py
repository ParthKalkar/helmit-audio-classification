import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

class DWConvBlock(nn.Module):
    def __init__(self, cin, cout, stride=(1,2), p=0.0):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, 3, stride=stride, padding=1, groups=cin, bias=False)
        self.dw_bn = nn.BatchNorm2d(cin)
        self.pw = nn.Conv2d(cin, cout, 1, bias=False)
        self.pw_bn = nn.BatchNorm2d(cout)
        self.act = nn.ReLU6(inplace=True)
        self.drop = nn.Dropout(p)
    def forward(self, x):
        x = self.act(self.dw_bn(self.dw(x)))
        x = self.act(self.pw_bn(self.pw(x)))
        return self.drop(x)

class LiteAudioCNN(nn.Module):
    def __init__(self, n_mels=128, n_classes=2, width_mult=0.6, dropout=0.3):
        super().__init__()
        c1 = int(32 * width_mult)
        c2 = int(64 * width_mult)
        c3 = int(128 * width_mult)
        c4 = int(160 * width_mult)
        c5 = int(192 * width_mult)

        self.stem = nn.Sequential(
            nn.Conv2d(1, c1, 3, stride=(2,2), padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU6(inplace=True)
        )
        self.block1 = DWConvBlock(c1, c2, stride=(1,2), p=dropout)
        self.block2 = DWConvBlock(c2, c3, stride=(2,2), p=dropout)
        self.block3 = DWConvBlock(c3, c4, stride=(2,2), p=dropout)
        self.block4 = DWConvBlock(c4, c5, stride=(2,2), p=dropout)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(c5, n_classes)
        )

    def forward(self, x, return_featmap=False):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        featmap = self.block4(x)
        out = self.head(featmap)
        if return_featmap:
            return out, featmap
        return out

class Wav2Vec2Classifier(nn.Module):
    def __init__(self, n_classes=2, freeze_base=True, num_unfrozen_layers=0):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        if freeze_base:
            for param in self.wav2vec2.parameters():
                param.requires_grad = False
        if num_unfrozen_layers > 0:
            # Unfreeze the last num_unfrozen_layers layers
            for layer in self.wav2vec2.encoder.layers[-num_unfrozen_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
        self.classifier = nn.Linear(self.wav2vec2.config.hidden_size, n_classes)

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs.last_hidden_state
        pooled = torch.mean(hidden_states, dim=1)  # Mean pooling over time
        logits = self.classifier(pooled)
        return logits
