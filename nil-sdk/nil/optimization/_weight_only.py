"""Explicit CPU FP32 inference with per-channel INT8 weight storage.

Weights are dequantized for the current linear operation. This reduces persistent
weight storage; it is not an INT8 activation kernel or a throughput guarantee.
"""
from nil.runtime import require
torch=require('torch','torch')


class WeightOnlyLinear(torch.nn.Module):
    def __init__(self,source):
        super().__init__()
        weight=source.weight.detach().cpu().float()
        if not torch.isfinite(weight).all():raise ValueError('Linear weights must be finite for INT8 storage')
        maximum=weight.abs().amax(dim=1)
        scale=torch.where(maximum>0,maximum/127,torch.ones_like(maximum))
        encoded=torch.round(weight/scale[:,None]).clamp(-127,127).to(torch.int8)
        self.in_features,self.out_features=source.in_features,source.out_features
        self.register_buffer('qweight',encoded)
        self.register_buffer('scale',scale)
        self.register_buffer('bias',source.bias.detach().cpu().float().clone() if source.bias is not None else None)
        self.eval()
    def forward(self,inputs):
        if inputs.device.type!='cpu' or inputs.dtype!=torch.float32:raise ValueError('Weight-only INT8 requires CPU float32 inputs')
        weight=self.qweight.to(torch.float32)*self.scale[:,None]
        return torch.nn.functional.linear(inputs,weight,self.bias)
    def extra_repr(self):return f'in_features={self.in_features}, out_features={self.out_features}, storage=int8-per-channel, compute=float32'


def convert(model,blocked):
    memo={}
    def visit(parent,prefix=''):
        for name,child in list(parent._modules.items()):
            if child is None:continue
            path=prefix+'.'+name if prefix else name
            if path in blocked:continue
            if type(child) is torch.nn.Linear:
                if id(child) not in memo:memo[id(child)]=WeightOnlyLinear(child)
                parent._modules[name]=memo[id(child)]
            else:visit(child,path)
    visit(model)
    return model
