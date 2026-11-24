"""Torch model definitions imported only by explicit graph construction."""
from nil.runtime import require
torch=require('torch','torch')


class GCN(torch.nn.Module):
    def __init__(self,inputs,hidden,outputs,task):
        super().__init__();self.first=torch.nn.Linear(inputs,hidden);self.second=torch.nn.Linear(hidden,outputs);self.task=task

    def extra_repr(self):return f'task={self.task}'

    def forward(self,features,edge_index,batch=None):
        from nil.graphs import propagate,pool
        hidden=self.first(propagate(features,edge_index)).relu()
        output=self.second(propagate(hidden,edge_index))
        if self.task=='graph':
            if batch is None:raise ValueError('Graph classification requires batch IDs')
            return pool(output,batch)
        if batch is not None:raise ValueError('Node classification does not accept batch IDs')
        return output
