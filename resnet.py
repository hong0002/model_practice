from ResNet.basic_block import BasicBlock
from ResNet.resnet_block import ResNet
import torch

model = ResNet(BasicBlock, [2, 2, 2, 2], 1000)
print(model)

# basic_block = BasicBlock(inplanes=64, planes=64)
# print(basic_block)

# x = torch.randn(1, 64, 56, 56)
# output = basic_block(x)
# print(output.shape)