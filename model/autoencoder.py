import torch
import torch.nn as nn

class Block(nn.Module):
    """
    U-Net block, consisting mainly of three convolutional filters with
    batch normalisation and ReLU activation functions.

    'down' blocks downsample at the output layer and outputs both downsampled
    and non-downsampled activations.

    'up' blocks concatenate non-downsampled corresponding feature maps and
    upsample at the output layer.

    'out' blocks concatenate non-downsampled corresponding feature maps and
    outputs the final feature maps, representing the final output layer of
    the model.
    """

    def __init__(self, in_channels, out_channels, direction='down'):
        assert direction in ['down', 'up', 'out'], "Direction must be either 'down', 'up' or 'out'."
        super(Block, self).__init__()
        if direction == 'down':
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
            self.out = nn.Conv2d(out_channels, out_channels, 2, 2, 0)
        else:
            self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, 3, 1, 1)
            self.conv2 = nn.ConvTranspose2d(out_channels, out_channels, 3, 1, 1)
            if direction == 'up':
                self.out = nn.ConvTranspose2d(out_channels, out_channels // 2, 2, 2, 0)
            elif direction == 'out':
                self.out = nn.ConvTranspose2d(out_channels, 1, 1, 1, 0)

        self.BN1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.BN2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()
        self.direction = direction

    def forward(self, inputs, cat_layers=None):
        if self.direction != 'down':
            assert cat_layers is not None, "'up' and 'out' directions must have concatenated layers."
            assert inputs.shape == cat_layers.shape, "Shape of both inputs and concatenated layers must be equal."
            inputs = torch.cat((inputs, cat_layers), dim=1)

        conv1 = self.conv1(inputs)
        BN1 = self.BN1(conv1)
        relu1 = self.relu1(BN1)
        conv2 = self.conv2(relu1)
        BN2 = self.BN2(conv2)
        relu2 = self.relu2(BN2)
        out = self.out(relu2)
        if self.direction == 'down':
            return out, relu2
        else:
            return out


class Encoder(nn.Module):
    """
    Encoder class, consists of three 'down' blocks.
    """

    def __init__(self):
        super(Encoder, self).__init__()
        self.block1 = Block(1, 16, 'down')
        self.block2 = Block(16, 32, 'down')

    def forward(self, inputs):
        block1, concat1 = self.block1(inputs)
        block2, concat2 = self.block2(block1)
        concats = [concat1, concat2]
        return block2, concats


class Decoder(nn.Module):
    """
    Decoder class, consists of two 'up' blocks and a final 'out' block.
    """

    def __init__(self):
        super(Decoder, self).__init__()
        self.block1 = Block(64, 32, 'up')
        self.block2 = Block(32, 16, 'out')

    def forward(self, inputs, concats):
        block1 = self.block1(inputs, concats[-1])
        block2 = self.block2(block1, concats[-2])
        return block2


class Autoencoder(nn.Module):
    """
    Autoencoder class, combines encoder and decoder model with a bottleneck
    layer in between.
    """

    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.bottleneck = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 2, 2, 0)
        )

    def forward(self, inputs):
        encoded, concatted = self.encoder(inputs)
        bottlenecked = self.bottleneck(encoded)
        decoded = self.decoder(bottlenecked, concatted)
        added = inputs + decoded
        return added