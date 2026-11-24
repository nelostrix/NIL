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


class _HeteroLayer(torch.nn.Module):
    def __init__(self,inputs,outputs,relations):
        super().__init__();self.types=tuple(sorted(inputs));self.relations=relations
        self.own=torch.nn.ModuleList([torch.nn.Linear(inputs[t],outputs[t]) for t in self.types])
        self.messages=torch.nn.ModuleList([torch.nn.Linear(inputs[a],outputs[b],bias=False) for a,_,b in relations])

    def forward(self,features,edges):
        from nil.graphs import relation_mean
        output={t:layer(features[t]) for t,layer in zip(self.types,self.own)}
        for relation,layer in zip(self.relations,self.messages):
            source,_,target=relation
            before=len(features[source])*layer.out_features
            after=len(features[target])*layer.in_features
            if min(before,after)>1000000:raise ValueError('Relation projection budget exceeds 1M')
            # A bias-free linear map commutes with mean aggregation. Choose the
            # smaller intermediate for rectangular bipartite relations.
            if before<=after:
                message=relation_mean(layer(features[source]),edges[relation],len(features[target]))
            else:
                message=layer(relation_mean(features[source],edges[relation],len(features[target])))
            output[target]=output[target]+message
        return output


class HeteroSAGE(torch.nn.Module):
    def __init__(self,inputs,hidden,outputs,relations):
        super().__init__();self.inputs=dict(sorted(inputs.items()));self.outputs=dict(sorted(outputs.items()));self.relations=relations;self.hidden=hidden
        widths={name:hidden for name in inputs}
        self.first=_HeteroLayer(inputs,widths,relations);self.second=_HeteroLayer(widths,outputs,relations)

    def extra_repr(self):return f'inputs={self.inputs},hidden={self.hidden},outputs={self.outputs},relations={self.relations}'

    def forward(self,features,edges):
        from nil.graphs import _hetero_validate
        _hetero_validate(features,edges)
        if set(features)!=set(self.inputs) or set(edges)!=set(self.relations) or any(features[k].shape[1]!=width for k,width in self.inputs.items()):raise ValueError('Typed model schema mismatch')
        if sum(len(x)*self.hidden for x in features.values())>1000000 or sum(len(features[t])*w for t,w in self.outputs.items())>1000000:raise ValueError('Typed activation budget exceeds 1M')
        hidden={key:value.relu() for key,value in self.first(features,edges).items()}
        return self.second(hidden,edges)
