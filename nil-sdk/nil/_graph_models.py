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


class WeightedGCN(GCN):
    def __init__(self,inputs,hidden,outputs,task,directed,normalization):
        super().__init__(inputs,hidden,outputs,task);self.directed=directed;self.normalization=normalization

    def extra_repr(self):return f'task={self.task},directed={self.directed},normalization={self.normalization}'

    def forward(self,features,edge_index,edge_weight,batch=None):
        from nil.graphs import weighted_propagate,pool
        options=dict(directed=self.directed,normalization=self.normalization)
        hidden=self.first(weighted_propagate(features,edge_index,edge_weight,**options)).relu()
        output=self.second(weighted_propagate(hidden,edge_index,edge_weight,**options))
        if self.task=='graph':
            if batch is None:raise ValueError('Graph classification requires batch IDs')
            return pool(output,batch)
        if batch is not None:raise ValueError('Node classification does not accept batch IDs')
        return output
