"""Local multi-process random-search coordination with durable, globally bounded trials."""
import hashlib
import json
import math
import random
import threading
import time
import uuid
from dataclasses import dataclass

from nil._classical_common import positive_int
from nil._ledger import connect


def _json(value): return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def _suggest(space,seed,index):
    rng=random.Random(f'{seed}:{index}'); parameters={}
    for name,spec in sorted(space.items()):
        if isinstance(spec,list): value=rng.choice(spec)
        elif spec['type']=='int':
            value=min(spec['high'],max(spec['low'],round(math.exp(rng.uniform(math.log(spec['low']),math.log(spec['high'])))))) if spec.get('log') else rng.randrange(spec['low'],spec['high']+1,spec.get('step',1))
        elif spec.get('log'): value=math.exp(rng.uniform(math.log(spec['low']),math.log(spec['high'])))
        elif 'step' in spec:
            count=math.floor((spec['high']-spec['low'])/spec['step']); value=spec['low']+rng.randrange(count+1)*spec['step']
        else: value=rng.uniform(spec['low'],spec['high'])
        if isinstance(value,float) and not math.isfinite(value): raise ValueError('Parameter range produced a nonfinite sample')
        parameters[name]=value
    return parameters


@dataclass(frozen=True)
class Trial:
    number: int
    parameters: dict
    token: str


class DistributedSearch:
    def __init__(self,path,*,space,max_trials,fingerprint,seed=0):
        from nil.selection import _space
        _space(space); positive_int(max_trials,'max_trials')
        if max_trials>10000 or len(space)>256 or len(_json(space))>65536: raise ValueError('Distributed search configuration exceeds limits')
        if not isinstance(fingerprint,str) or not fingerprint: raise ValueError('A workload fingerprint is required')
        if type(seed) is not int or seed<0: raise ValueError('seed must be a nonnegative integer')
        params=[_json(_suggest(space,seed,index)) for index in range(max_trials)]
        self.path,self.fingerprint=str(path),fingerprint; self.db=connect(path)
        self.db.executescript('''
CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS trials(number INTEGER PRIMARY KEY,params TEXT NOT NULL,status TEXT NOT NULL,owner TEXT,token TEXT,expires REAL,score REAL,error TEXT);
''')
        definition=_json({'schema_version':1,'space':space,'max_trials':max_trials,'seed':seed,'fingerprint':fingerprint,'sampler':'index-seeded-random-v1'})
        self.db.execute('BEGIN IMMEDIATE')
        try:
            previous=self.db.execute("SELECT value FROM metadata WHERE key='definition'").fetchone()
            if previous and previous[0]!=definition: raise ValueError('Search workload/configuration differs from persisted study')
            if not previous:
                self.db.execute("INSERT INTO metadata VALUES('definition',?)",(definition,))
                self.db.executemany("INSERT INTO trials(number,params,status) VALUES(?,?,'pending')",enumerate(params))
            self.db.execute('COMMIT')
        except BaseException:
            self.db.execute('ROLLBACK'); self.db.close(); raise

    def claim(self,worker,*,lease_seconds=60):
        if not isinstance(worker,str) or not worker or len(worker)>128: raise ValueError('worker needs 1–128 characters')
        if isinstance(lease_seconds,bool) or not isinstance(lease_seconds,(int,float)) or not math.isfinite(lease_seconds) or not .1<=lease_seconds<=86400: raise ValueError('lease_seconds must be .1–86400')
        self.db.execute('BEGIN IMMEDIATE')
        try:
            now=time.time()
            # Expired attempts consume their original slot; never rerun an unknown
            # amount of work and silently exceed the physical fit budget.
            self.db.execute("UPDATE trials SET status='failed',error='Worker lease expired',token=NULL WHERE status='running' AND expires<?",(now,))
            row=self.db.execute("SELECT number,params FROM trials WHERE status='pending' ORDER BY number LIMIT 1").fetchone()
            if row is None: self.db.execute('COMMIT'); return None
            token=uuid.uuid4().hex
            self.db.execute("UPDATE trials SET status='running',owner=?,token=?,expires=? WHERE number=?",(worker,token,now+lease_seconds,row[0]))
            self.db.execute('COMMIT'); return Trial(row[0],json.loads(row[1]),token)
        except BaseException: self.db.execute('ROLLBACK'); raise

    def heartbeat(self,trial,*,lease_seconds=60):
        if not math.isfinite(lease_seconds) or lease_seconds<.1 or lease_seconds>86400: raise ValueError('Invalid lease duration')
        updated=self.db.execute("UPDATE trials SET expires=? WHERE number=? AND token=? AND status='running' AND expires>=?",(time.time()+lease_seconds,trial.number,trial.token,time.time()))
        if not updated.rowcount: raise RuntimeError('Trial lease lost; result cannot be committed')

    def complete(self,trial,score):
        if isinstance(score,bool) or not isinstance(score,(int,float)) or not math.isfinite(score): raise ValueError('Score must be finite')
        self._finish(trial,'completed',score,None)

    def fail(self,trial,error): self._finish(trial,'failed',None,str(error)[:4000])

    def _finish(self,trial,status,score,error):
        updated=self.db.execute('UPDATE trials SET status=?,score=?,error=?,token=NULL WHERE number=? AND token=? AND status=\'running\' AND expires>=?',(status,score,error,trial.number,trial.token,time.time()))
        if not updated.rowcount: raise RuntimeError('Trial lease lost; result cannot be committed')

    def results(self):
        return [{'number':n,'params':json.loads(params),'status':state,'score':score,'error':error} for n,params,state,score,error in self.db.execute('SELECT number,params,status,score,error FROM trials ORDER BY number')]

    def best(self):
        completed=[trial for trial in self.results() if trial['status']=='completed']
        if not completed: raise ValueError('No completed trial is available')
        return max(completed,key=lambda trial:(trial['score'],-trial['number']))

    def run(self,estimator,features,targets,*,cv_plan,scoring,groups=None,sample_weight=None,worker='worker',lease_seconds=60):
        from nil.runtime import require
        from nil.validation import cross_validate
        from nil.selection import _distributed_fingerprint
        if _distributed_fingerprint(estimator,features,targets,cv_plan,groups,sample_weight,scoring)!=self.fingerprint:
            raise ValueError("Worker workload differs from the persisted search")
        base=require('sklearn.base','classical')
        while (trial:=self.claim(worker,lease_seconds=lease_seconds)) is not None:
            stop=threading.Event(); lost=[]
            def renew(trial=trial,stop=stop,lost=lost):
                db=connect(self.path)
                try:
                    while not stop.wait(max(.02,lease_seconds/3)):
                        now=time.time()
                        update=db.execute("UPDATE trials SET expires=? WHERE number=? AND token=? AND status='running' AND expires>=?",(now+lease_seconds,trial.number,trial.token,now))
                        if not update.rowcount: lost.append('Trial lease lost'); return
                except Exception as error: lost.append(str(error))
                finally: db.close()
            thread=threading.Thread(target=renew,daemon=True); thread.start()
            try:
                result=cross_validate(base.clone(estimator).set_params(**trial.parameters),features,targets,
                    cv=cv_plan,scoring=scoring,groups=groups,sample_weight=sample_weight,n_jobs=1,max_fits=len(cv_plan))
                score=next(iter(result.summary.values()))['mean']
                if lost: raise RuntimeError(lost[0])
                self.complete(trial,float(score))
            except Exception as error:
                try: self.fail(trial,error)
                except RuntimeError: pass
                if isinstance(error,(MemoryError,RecursionError)): raise
            finally: stop.set(); thread.join(timeout=6)
        return self.results()

    def close(self): self.db.close()
    def __enter__(self): return self
    def __exit__(self,*_): self.close()
