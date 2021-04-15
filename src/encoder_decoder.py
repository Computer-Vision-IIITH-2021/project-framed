import torch
import torch.nn as nn
import numpy as np

from harr import WavePool,WaveUnpool

class WaveEncoder(nn.Module):
    def __init__(self, option_unpool):
        super(WaveEncoder, self).__init__()
        self.option_unpool = option_unpool

        self.pad = nn.ReflectionPad2d(1)
        self.relu = nn.ReLU(inplace=True)

        self.conv0 = nn.Conv2d(3, 3, 1, 1, 0)
        self.conv1_1 = nn.Conv2d(3, 64, 3, 1, 0)
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 0)
        self.pool1 = WavePool(64)

        self.conv2_1 = nn.Conv2d(64, 128, 3, 1, 0)
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 0)
        self.pool2 = WavePool(128)

        self.conv3_1 = nn.Conv2d(128, 256, 3, 1, 0)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv3_4 = nn.Conv2d(256, 256, 3, 1, 0)
        self.pool3 = WavePool(256)

        self.conv4_1 = nn.Conv2d(256, 512, 3, 1, 0)

    def forward(self, x):
        skips = {}
        for level in [1, 2, 3, 4]:
            x = self.encode(x, skips, level)
        return x

    def encode(self, x, skips, level):
        assert level in {1, 2, 3, 4}
        if self.option_unpool == 'sum':
            if level == 1:
                out = self.conv0(x)
                out = self.relu(self.conv1_1(self.pad(out)))
                out = self.relu(self.conv1_2(self.pad(out)))
                skips['conv1_2'] = out
                LL, LH, HL, HH = self.pool1(out)
                skips['pool1'] = [LH, HL, HH]
                return LL
            elif level == 2:
                out = self.relu(self.conv2_1(self.pad(x)))
                out = self.relu(self.conv2_2(self.pad(out)))
                skips['conv2_2'] = out
                LL, LH, HL, HH = self.pool2(out)
                skips['pool2'] = [LH, HL, HH]
                return LL
            elif level == 3:
                out = self.relu(self.conv3_1(self.pad(x)))
                out = self.relu(self.conv3_2(self.pad(out)))
                out = self.relu(self.conv3_3(self.pad(out)))
                out = self.relu(self.conv3_4(self.pad(out)))
                skips['conv3_4'] = out
                LL, LH, HL, HH = self.pool3(out)
                skips['pool3'] = [LH, HL, HH]
                return LL
            else:
                return self.relu(self.conv4_1(self.pad(x)))

        elif self.option_unpool == 'cat5':
            if level == 1:
                out = self.conv0(x)
                out = self.relu(self.conv1_1(self.pad(out)))
                return out

            elif level == 2:
                out = self.relu(self.conv1_2(self.pad(x)))
                skips['conv1_2'] = out
                LL, LH, HL, HH = self.pool1(out)
                skips['pool1'] = [LH, HL, HH]
                out = self.relu(self.conv2_1(self.pad(LL)))
                return out

            elif level == 3:
                out = self.relu(self.conv2_2(self.pad(x)))
                skips['conv2_2'] = out
                LL, LH, HL, HH = self.pool2(out)
                skips['pool2'] = [LH, HL, HH]
                out = self.relu(self.conv3_1(self.pad(LL)))
                return out

            else:
                out = self.relu(self.conv3_2(self.pad(x)))
                out = self.relu(self.conv3_3(self.pad(out)))
                out = self.relu(self.conv3_4(self.pad(out)))
                skips['conv3_4'] = out
                LL, LH, HL, HH = self.pool3(out)
                skips['pool3'] = [LH, HL, HH]
                out = self.relu(self.conv4_1(self.pad(LL)))
                return out
        else:
            raise NotImplementedError


class WaveDecoder(nn.Module):
    def __init__(self, option_unpool):
        super(WaveDecoder, self).__init__()
        self.option_unpool = option_unpool

        if option_unpool == 'sum':
            multiply_in = 1
        elif option_unpool == 'cat5':
            multiply_in = 5
        else:
            raise NotImplementedError

        self.pad = nn.ReflectionPad2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.conv4_1 = nn.Conv2d(512, 256, 3, 1, 0)

        self.recon_block3 = WaveUnpool(256, option_unpool)
        if option_unpool == 'sum':
            self.conv3_4 = nn.Conv2d(256*multiply_in, 256, 3, 1, 0)
        else:
            self.conv3_4_2 = nn.Conv2d(256*multiply_in, 256, 3, 1, 0)
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv3_1 = nn.Conv2d(256, 128, 3, 1, 0)

        self.recon_block2 = WaveUnpool(128, option_unpool)
        if option_unpool == 'sum':
            self.conv2_2 = nn.Conv2d(128*multiply_in, 128, 3, 1, 0)
        else:
            self.conv2_2_2 = nn.Conv2d(128*multiply_in, 128, 3, 1, 0)
        self.conv2_1 = nn.Conv2d(128, 64, 3, 1, 0)

        self.recon_block1 = WaveUnpool(64, option_unpool)
        if option_unpool == 'sum':
            self.conv1_2 = nn.Conv2d(64*multiply_in, 64, 3, 1, 0)
        else:
            self.conv1_2_2 = nn.Conv2d(64*multiply_in, 64, 3, 1, 0)
        self.conv1_1 = nn.Conv2d(64, 3, 3, 1, 0)

    def forward(self, x, skips):
        for level in [4, 3, 2, 1]:
            x = self.decode(x, skips, level)
        return x

    def decode(self, x, skips, level):
        assert level in {4, 3, 2, 1}
        if level == 4:
            out = self.relu(self.conv4_1(self.pad(x)))
            LH, HL, HH = skips['pool3']
            original = skips['conv3_4'] if 'conv3_4' in skips.keys() else None
            out = self.recon_block3(out, LH, HL, HH, original)
            _conv3_4 = self.conv3_4 if self.option_unpool == 'sum' else self.conv3_4_2
            out = self.relu(_conv3_4(self.pad(out)))
            out = self.relu(self.conv3_3(self.pad(out)))
            return self.relu(self.conv3_2(self.pad(out)))
        elif level == 3:
            out = self.relu(self.conv3_1(self.pad(x)))
            LH, HL, HH = skips['pool2']
            original = skips['conv2_2'] if 'conv2_2' in skips.keys() else None
            out = self.recon_block2(out, LH, HL, HH, original)
            _conv2_2 = self.conv2_2 if self.option_unpool == 'sum' else self.conv2_2_2
            return self.relu(_conv2_2(self.pad(out)))
        elif level == 2:
            out = self.relu(self.conv2_1(self.pad(x)))
            LH, HL, HH = skips['pool1']
            original = skips['conv1_2'] if 'conv1_2' in skips.keys() else None
            out = self.recon_block1(out, LH, HL, HH, original)
            _conv1_2 = self.conv1_2 if self.option_unpool == 'sum' else self.conv1_2_2
            return self.relu(_conv1_2(self.pad(out)))
        else:
            return self.conv1_1(self.pad(x))
