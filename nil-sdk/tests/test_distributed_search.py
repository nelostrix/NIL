import json,os,subprocess,sys,time,multiprocessing
from pathlib import Path
import pytest
pytest.importorskip('sklearn')
import numpy as np
from sklearn.linear_model import Ridge
import nil


SCRIPT='''
import sys,numpy as np
from sklearn.linear_model import Ridge
from nil.selection import distributed_search
x=np.arange(40,dtype=float).reshape(-1,1)/10; y=x[:,0]*2
with distributed_search(Ridge(),x,y,{'alpha':[.01,.1,1.,10.]},storage=sys.argv[1],max_trials=8,cv=2,max_fits=16) as search:
 search.run(Ridge(),x,y,cv_plan=search.cv_plan,scoring=search.scoring,worker=sys.argv[2],lease_seconds=2)
'''


def _worker(path,name):
    sys.argv=['worker',str(path),name]
    exec(SCRIPT, {'__name__':'__main__'})


def create(path,**options):
    x=np.arange(40,dtype=float).reshape(-1,1)/10; y=x[:,0]*2
    return nil.selection.distributed_search(Ridge(),x,y,{'alpha':[.01,.1,1.,10.]},storage=path,max_trials=8,cv=2,max_fits=16,**options)


def test_real_multi_process_global_budget_and_sampler_restart(tmp_path):
    path=tmp_path/'study.db'
    with create(path) as search: expected=[trial['params'] for trial in search.results()]
    method='forkserver' if 'forkserver' in multiprocessing.get_all_start_methods() else 'spawn'
    if method=='forkserver': multiprocessing.set_forkserver_preload(['numpy','sklearn.linear_model','nil.selection'])
    context=multiprocessing.get_context(method)
    children=[context.Process(target=_worker,args=(path,f'worker-{i}')) for i in range(2)]
    for child in children: child.start()
    try:
        for child in children:
            child.join(timeout=30)
            assert child.exitcode==0
        with create(path) as resumed:
            trials=resumed.results()
            assert len(trials)==8 and all(trial['status']=='completed' for trial in trials)
            assert [trial['params'] for trial in trials]==expected
            assert resumed.claim('extra') is None
            assert resumed.best()['score']<=0
        with create(tmp_path/'repeat.db') as repeat: assert [trial['params'] for trial in repeat.results()]==expected
        with pytest.raises(ValueError,match='differs'): create(path,seed=1)
    finally:
        for child in children:
            if child.is_alive(): child.kill(); child.join()


def test_actual_dead_worker_expires_consumes_budget_and_rejects_old_token(tmp_path):
    path=tmp_path/'lease.db'; ready=tmp_path/'claimed.json'
    code='''
import sys,json,time
from pathlib import Path
from nil._distributed_search import DistributedSearch
with DistributedSearch(sys.argv[1],space={'alpha':[1,2]},max_trials=2,fingerprint='fixed') as search:
 trial=search.claim('dead',lease_seconds=.2)
 Path(sys.argv[2]).write_text(json.dumps(vars(trial)))
 time.sleep(60)
'''
    child=subprocess.Popen([sys.executable,'-c',code,str(path),str(ready)])
    try:
        deadline=time.monotonic()+5
        while not ready.exists() and time.monotonic()<deadline: time.sleep(.02)
        assert ready.exists(); child.kill(); child.wait(timeout=5)
        time.sleep(.25)
        from nil._distributed_search import DistributedSearch,Trial
        with DistributedSearch(path,space={'alpha':[1,2]},max_trials=2,fingerprint='fixed') as search:
            next_trial=search.claim('replacement')
            assert next_trial.number==1
            old=Trial(**json.loads(ready.read_text()))
            with pytest.raises(RuntimeError,match='lease lost'): search.complete(old,1.)
            search.complete(next_trial,.5)
            assert search.claim('extra') is None
            assert [trial['status'] for trial in search.results()]==['failed','completed']
    finally:
        if child.poll() is None: child.kill(); child.wait()


def test_worker_cannot_publish_scores_for_a_different_dataset(tmp_path):
    x=np.arange(40,dtype=float).reshape(-1,1)/10; y=x[:,0]*2
    with create(tmp_path/'study.db') as search:
        with pytest.raises(ValueError,match='workload differs'):
            search.run(Ridge(),x,y+100,cv_plan=search.cv_plan,scoring=search.scoring)
        assert all(trial['status']=='pending' for trial in search.results())
