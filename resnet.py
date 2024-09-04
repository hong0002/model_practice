from ResNet.basic_block import BasicBlock
import torch

basic_block = BasicBlock(inplanes=64, planes=64)
print(basic_block)

x = torch.randn(1, 64, 56, 56)
output = basic_block(x)
print(output.shape)