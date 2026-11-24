"""Explicit bounded COO/CSR matrices; never implicitly densify input."""
import json
import math
from pathlib import Path
from ._native_tensor import Tensor, MAX_ELEMENTS


def _shape(shape):
    if not isinstance(shape,(tuple,list)) or len(shape)!=2 or any(type(n) is not int or not 0<n<=2147483647 for n in shape):
        raise ValueError('Sparse shape needs two positive integer dimensions <=2**31-1')
    return tuple(shape)


def _values(values,count):
    if isinstance(values,Tensor):
        if values.shape not in ((count,),(1,count)):raise ValueError('Sparse values must be a vector matching nnz')
        return values
    if not isinstance(values,(list,tuple)) or len(values)!=count:raise ValueError('Sparse value count differs')
    return Tensor(values,shape=(count,)) if count else None


class SparseCOO:
    def __init__(self,indices,values,shape):
        self._shape=_shape(shape)
        if not isinstance(indices,(list,tuple)) or len(indices)>MAX_ELEMENTS:raise ValueError('Sparse nnz exceeds 100,000')
        pairs=[]
        for pair in indices:
            if not isinstance(pair,(list,tuple)) or len(pair)!=2 or any(type(i) is not int or not 0<=i<size for i,size in zip(pair,self.shape)):
                raise ValueError('Sparse coordinate out of bounds')
            pairs.append(tuple(pair))
        self._indices=tuple(pairs);self._values=_values(values,len(pairs))
    @property
    def shape(self):return self._shape
    @property
    def indices(self):return self._indices
    @property
    def values(self):return self._values
    @property
    def nnz(self):return len(self.indices)

    def coalesce(self):
        if not self.nnz:return SparseCOO([],[],self.shape)
        pairs=tuple(sorted(set(self.indices)));positions={pair:i for i,pair in enumerate(pairs)}
        mapping=[positions[pair] for pair in self.indices];values=[0.]*len(pairs)
        for index,value in zip(mapping,self.values._flat):values[index]+=value
        parent=self.values
        merged=Tensor._op_flat(values,(len(pairs),),(parent,),lambda g:[(parent,[g[i] for i in mapping])])
        return SparseCOO(pairs,merged,self.shape)

    def to_csr(self):
        if self.shape[0]+1>MAX_ELEMENTS:raise ValueError('CSR row pointer exceeds reference storage limit')
        canonical=self.coalesce();pointer=[0]*(self.shape[0]+1)
        for row,_ in canonical.indices:pointer[row+1]+=1
        for i in range(self.shape[0]):pointer[i+1]+=pointer[i]
        return SparseCSR(pointer,[column for _,column in canonical.indices],canonical.values if canonical.nnz else [],self.shape)

    def to_dense(self):
        if math.prod(self.shape)>MAX_ELEMENTS:raise ValueError('Explicit densification exceeds 100,000-element limit')
        values=[0.]*math.prod(self.shape)
        for (row,column),value in zip(self.indices,self.values._flat if self.values else ()):
            values[row*self.shape[1]+column]+=value
        parents=(self.values,) if self.nnz else ()
        parent=self.values
        return Tensor._op_flat(values,self.shape,parents,lambda g:[(parent,[g[row*self.shape[1]+column] for row,column in self.indices])] if parent else [])

    def matmul(self,dense):
        if not isinstance(dense,Tensor) or dense.ndim!=2 or dense.shape[0]!=self.shape[1]:raise ValueError('Sparse matmul requires a compatible rank2 dense Tensor')
        rows,columns=self.shape[0],dense.shape[1]
        if rows*columns>MAX_ELEMENTS:raise ValueError('Sparse matmul output exceeds 100,000-element limit')
        result=[0.]*(rows*columns);a=self.values._flat if self.nnz else ();b=dense._flat
        for (row,column),value in zip(self.indices,a):
            for j in range(columns):result[row*columns+j]+=value*b[column*columns+j]
        parent=self.values
        def backward(g):
            ga=[];gb=[0.]*dense.numel()
            for (row,column),value in zip(self.indices,a):
                ga.append(sum(g[row*columns+j]*b[column*columns+j] for j in range(columns)))
                for j in range(columns):gb[column*columns+j]+=g[row*columns+j]*value
            return [(dense,gb)]+([(parent,ga)] if parent else [])
        return Tensor._op_flat(result,(rows,columns),(dense,parent) if parent else (dense,),backward)
    __matmul__=matmul

    def state_dict(self):
        return {'format':'nil-native-coo-v1','shape':list(self.shape),'indices':[list(pair) for pair in self.indices],
                'values':list(self.values._flat) if self.nnz else []}


class SparseCSR:
    def __init__(self,indptr,indices,values,shape):
        self._shape=_shape(shape)
        if not isinstance(indptr,(list,tuple)) or len(indptr)!=self.shape[0]+1 or len(indptr)>MAX_ELEMENTS:
            raise ValueError('Invalid CSR row pointer length/budget')
        if not isinstance(indices,(list,tuple)) or len(indices)>MAX_ELEMENTS or any(type(i) is not int or not 0<=i<self.shape[1] for i in indices):raise ValueError('Invalid CSR column indices')
        if any(type(i) is not int or not 0<=i<=len(indices) for i in indptr) or indptr[0]!=0 or indptr[-1]!=len(indices) or any(a>b for a,b in zip(indptr,indptr[1:])):
            raise ValueError('CSR row pointers must be monotone and span nnz')
        self._indptr,self._indices=tuple(indptr),tuple(indices);self._values=_values(values,len(indices))
    @property
    def shape(self):return self._shape
    @property
    def indptr(self):return self._indptr
    @property
    def indices(self):return self._indices
    @property
    def values(self):return self._values
    @property
    def nnz(self):return len(self.indices)
    def to_coo(self):
        pairs=[(row,self.indices[index]) for row in range(self.shape[0]) for index in range(self.indptr[row],self.indptr[row+1])]
        return SparseCOO(pairs,self.values if self.nnz else [],self.shape)
    def coalesce(self):return self.to_coo().to_csr()
    def to_dense(self):return self.to_coo().to_dense()
    def matmul(self,dense):return self.to_coo().matmul(dense)
    __matmul__=matmul
    def state_dict(self):
        return {'format':'nil-native-csr-v1','shape':list(self.shape),'indptr':list(self.indptr),
                'indices':list(self.indices),'values':list(self.values._flat) if self.nnz else []}


def save_sparse(matrix,path):
    if not isinstance(matrix,(SparseCOO,SparseCSR)):raise TypeError('Expected native COO/CSR')
    text=json.dumps(matrix.state_dict(),allow_nan=False)
    if len(text.encode())>16000000:raise ValueError('Sparse artifact exceeds 16 MB')
    with Path(path).open('x',encoding='utf8') as stream:stream.write(text)
    return Path(path)


def load_sparse(path):
    path=Path(path)
    if path.is_symlink() or path.stat().st_size>16000000:raise ValueError('Invalid sparse artifact')
    state=json.loads(path.read_text())
    if not isinstance(state,dict):raise ValueError('Invalid sparse artifact')
    if state.get('format')=='nil-native-coo-v1' and set(state)=={'format','shape','indices','values'}:
        return SparseCOO(state['indices'],state['values'],state['shape'])
    if state.get('format')=='nil-native-csr-v1' and set(state)=={'format','shape','indptr','indices','values'}:
        return SparseCSR(state['indptr'],state['indices'],state['values'],state['shape'])
    raise ValueError('Unsupported sparse artifact')
