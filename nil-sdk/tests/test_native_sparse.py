"""Sparse correctness, duplicate gradients, bounded storage and JSON roundtrips."""
import tracemalloc
import pytest
from nil import native as n


def test_coo_duplicates_csr_canonicalization_and_explicit_densification(tmp_path):
    matrix=n.SparseCOO([(1,0),(0,1),(0,1)],[3.,2.,-1.],(2,3))
    assert matrix.to_dense().tolist()==[[0.,1.,0.],[3.,0.,0.]]
    canonical=matrix.coalesce();assert canonical.indices==((0,1),(1,0)) and canonical.values.tolist()==[1.,3.]
    csr=matrix.to_csr();assert csr.indptr==(0,1,2) and csr.indices==(1,0)
    assert csr.to_coo().to_dense().tolist()==matrix.to_dense().tolist()
    for value in (matrix,csr):
        restored=n.load_sparse(n.save_sparse(value,tmp_path/(type(value).__name__+'.json')))
        assert restored.state_dict()==value.state_dict()
    empty=n.SparseCSR([0,0,0],[],[],(2,3))
    dense=n.Tensor([[1,2],[3,4],[5,6]],requires_grad=True)
    assert (empty@dense).tolist()==[[0.,0.],[0.,0.]]
    (empty@dense).sum().backward();assert dense.grad==((0.,0.),(0.,0.),(0.,0.))


@pytest.mark.parametrize('layout',['coo','coalesced','csr','dense'])
def test_sparse_value_and_dense_gradients_match_independent_reference(layout):
    values=n.Tensor([.4,-.2,.7],shape=(3,),requires_grad=True)
    dense=n.Tensor([[.3,-.8],[.2,.6],[-.4,.1]],requires_grad=True)
    matrix=n.SparseCOO([(0,1),(0,1),(1,0)],values,(2,3))
    if layout=='coalesced':matrix=matrix.coalesce()
    if layout=='csr':matrix=matrix.to_csr()
    if layout=='dense':matrix=matrix.to_dense()
    loss=((matrix@dense)**2).mean();loss.backward()
    def reference(v,b):return sum(x*x for row in [[(v[0]+v[1])*b[1][j] for j in range(2)],[v[2]*b[0][j] for j in range(2)]] for x in row)/4
    for i in range(3):
        plus=values.tolist();minus=values.tolist();plus[i]+=1e-6;minus[i]-=1e-6
        assert values.grad[i]==pytest.approx((reference(plus,dense.tolist())-reference(minus,dense.tolist()))/2e-6,abs=1e-8)
    for i in range(3):
        for j in range(2):
            plus=dense.tolist();minus=dense.tolist();plus[i][j]+=1e-6;minus[i][j]-=1e-6
            assert dense.grad[i][j]==pytest.approx((reference(values.tolist(),plus)-reference(values.tolist(),minus))/2e-6,abs=1e-8)


def test_sparse_no_implicit_dense_allocation_and_version_safety():
    tracemalloc.start()
    try:
        huge=n.SparseCOO([(999999999,999999999)],[1.],(1000000000,1000000000))
        assert huge.nnz==1
        with pytest.raises(ValueError,match='densification'):huge.to_dense()
        with pytest.raises(ValueError,match='row pointer'):huge.to_csr()
        assert tracemalloc.get_traced_memory()[1]<100000
    finally:tracemalloc.stop()
    values=n.Tensor([1.],shape=(1,),requires_grad=True)
    matrix=n.SparseCOO([(0,0)],values,(1,1));loss=(matrix@n.Tensor([[2.]])).sum()
    values.assign([3.])
    with pytest.raises(RuntimeError,match='changed'):loss.backward()
    assert values.grad is None


@pytest.mark.parametrize('build',[
    lambda:n.SparseCOO([(True,0)],[1],(2,2)),lambda:n.SparseCOO([(2,0)],[1],(2,2)),
    lambda:n.SparseCOO([(0,0)],[float('nan')],(2,2)),lambda:n.SparseCOO([],[],(0,2)),
    lambda:n.SparseCSR([0,2,1],[0],[1],(2,2)),lambda:n.SparseCSR([0,0,2],[0],[1],(2,2)),
    lambda:n.SparseCSR([0,1],[3],[1],(1,2)),
])
def test_invalid_sparse_inputs_fail_explicitly(build):
    with pytest.raises(ValueError):build()


def test_sparse_matches_torch_double_precision():
    torch=pytest.importorskip('torch')
    v=n.Tensor([.4,-.2,.7],shape=(3,),requires_grad=True);b=n.Tensor([[.3],[.8],[.1]],requires_grad=True)
    sparse=n.SparseCOO([(0,1),(0,1),(1,0)],v,(2,3)).to_csr()
    result=sparse@b;(result**2).mean().backward()
    tv=torch.tensor(v.tolist(),dtype=torch.float64,requires_grad=True);tb=torch.tensor(b.tolist(),dtype=torch.float64,requires_grad=True)
    reference=torch.sparse_coo_tensor(torch.tensor([[0,0,1],[1,1,0]]),tv,(2,3)).to_dense()@tb
    reference.square().mean().backward()
    torch.testing.assert_close(torch.tensor(result.tolist(),dtype=torch.float64),reference)
    torch.testing.assert_close(torch.tensor(v.grad,dtype=torch.float64),tv.grad)
    torch.testing.assert_close(torch.tensor(b.grad,dtype=torch.float64),tb.grad)
