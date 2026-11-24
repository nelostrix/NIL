"""Actual local compilation and independent matmul/VJP parity, no numerical runtime."""
import concurrent.futures
import random
import shutil
import sys
import pytest
from nil import native as n
from nil._native_tensor import unpack


@pytest.fixture(scope='module')
def kernel(tmp_path_factory):
    if sys.platform!='linux' or not (shutil.which('cc') or shutil.which('clang')):
        pytest.skip('Explicit Linux C compiler acceptance target required')
    directory=tmp_path_factory.mktemp('nil-compiled-kernel')
    binary=n.build_cpu_kernel(directory)
    with pytest.raises(FileExistsError):n.build_cpu_kernel(directory)
    return n.load_cpu_kernel(binary)


@pytest.mark.parametrize('shape',[(3,4,2),(1,7,1),(5,1,3)])
def test_actual_compiled_forward_backward_matches_python_float64(kernel,shape):
    m,k,columns=shape;rng=random.Random(7)
    av=[rng.uniform(-1,1) for _ in range(m*k)];bv=[rng.uniform(-1,1) for _ in range(k*columns)]
    left=n.Tensor(av,shape=(m,k),requires_grad=True);right=n.Tensor(bv,shape=(k,columns),requires_grad=True)
    a=n.Tensor(av,shape=(m,k),requires_grad=True);b=n.Tensor(bv,shape=(k,columns),requires_grad=True)
    result=left.matmul(right,kernel=kernel);expected=a@b
    assert result._flat==pytest.approx(expected._flat,rel=1e-12,abs=1e-12)
    (result**2).mean().backward();(expected**2).mean().backward()
    assert unpack(left.grad)[1]==pytest.approx(unpack(a.grad)[1],rel=1e-12,abs=1e-12)
    assert unpack(right.grad)[1]==pytest.approx(unpack(b.grad)[1],rel=1e-12,abs=1e-12)


def test_compiled_batched_matmul_and_concurrent_calls(kernel):
    a=n.Tensor(list(range(24)),shape=(2,3,4),requires_grad=True);b=n.Tensor([.1]*8,shape=(4,2),requires_grad=True)
    assert a.matmul(b,kernel=kernel)._flat==pytest.approx((a@b)._flat,rel=1e-12,abs=1e-12)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        outputs=list(pool.map(lambda _:kernel.matmul([1.,2.],[3.,4.],1,2,1),range(16)))
    assert outputs==[[11.]]*16


@pytest.mark.parametrize('a,b,m,k,columns',[
    ([1.],[1.],0,1,1),([1.],[1.],1,100001,1),([1.],[1.],True,1,1),
    ([1.],[1.],2,2,2),([float('nan')],[1.],1,1,1),([1e308],[1e308],1,1,1),
])
def test_native_kernel_refuses_invalid_dimensions_values_and_overflow(kernel,a,b,m,k,columns):
    with pytest.raises((ValueError,OverflowError)):kernel.matmul(a,b,m,k,columns)


def test_kernel_loader_refuses_foreign_abi_or_changed_binary_metadata(kernel,tmp_path):
    from pathlib import Path
    import json
    source=Path(kernel.__file__);target=tmp_path/source.name
    shutil.copy2(source,target)
    metadata=json.loads(source.with_suffix(source.suffix+'.json').read_text())
    sidecar=target.with_suffix(target.suffix+'.json')
    sidecar.write_text(json.dumps({**metadata,'abi':'foreign-python-abi'}))
    with pytest.raises(ValueError,match='ABI'):n.load_cpu_kernel(target)
    sidecar.write_text(json.dumps({**metadata,'binary_sha256':'0'*64}))
    with pytest.raises(ValueError,match='digest'):n.load_cpu_kernel(target)
