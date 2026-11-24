"""Bounded dense/hybrid indexing with caller-supplied or local native encoders."""
import base64
import hashlib
import math
from nil.rag import Index
from nil.runtime import require
from nil._workflow_io import save as _save,load as _load


class SemanticIndex(Index):
    def __init__(self,encoder,*,mode='hybrid',min_similarity=.2,lexical_weight=.5,rrf_k=60,reranker=None,reranker_id=None,**config):
        if not callable(getattr(encoder,'encode',None)) or not isinstance(getattr(encoder,'fingerprint',None),str) or not encoder.fingerprint:raise TypeError('Encoder requires encode(texts) and an explicit fingerprint')
        if mode not in {'dense','hybrid'} or type(min_similarity) not in (int,float) or not -1<=min_similarity<=1 or type(lexical_weight) not in (int,float) or not 0<=lexical_weight<=1 or type(rrf_k) is not int or not 1<=rrf_k<=1000:raise ValueError('Invalid semantic ranking config')
        if (reranker is None)!=(reranker_id is None) or reranker is not None and (not callable(reranker) or not isinstance(reranker_id,str) or not reranker_id):raise ValueError('Reranker requires a callable and explicit identity')
        super().__init__(**config)
        self.encoder=encoder;self.encoder_id=encoder.fingerprint;self.reranker=reranker
        self.ranking=dict(mode=mode,min_similarity=min_similarity,lexical_weight=lexical_weight,rrf_k=rrf_k,reranker_id=reranker_id)
        self._vectors=None

    def _check_encoder(self):
        if self.encoder.fingerprint!=self.encoder_id:raise ValueError('Encoder identity changed; explicitly rebuild the index')

    def _rebuild(self,documents):
        np=require('numpy','audio');self._check_encoder()
        candidate=Index(**self.config);Index._rebuild(candidate,documents)
        if not candidate._chunks:
            vectors=np.empty((0,0),dtype=np.float32)
        else:
            old={c['id']:(c['text'],self._vectors[i]) for i,c in enumerate(self._chunks)} if self._vectors is not None else {}
            new=[c for c in candidate._chunks if c['id'] not in old or old[c['id']][0]!=c['text']]
            fresh={}
            if new:
                encoded=np.asarray(self.encoder.encode([c['text'] for c in new]),dtype=np.float32)
                if encoded.ndim!=2 or encoded.shape[0]!=len(new) or not 1<=encoded.shape[1]<=4096 or not np.isfinite(encoded).all() or not np.isfinite(np.linalg.norm(encoded,axis=1)).all() or (np.linalg.norm(encoded,axis=1)==0).any():raise ValueError('Invalid encoder output')
                encoded=encoded/np.linalg.norm(encoded,axis=1,keepdims=True)
                fresh={c['id']:encoded[i] for i,c in enumerate(new)}
            vectors=np.stack([fresh[c['id']] if c['id'] in fresh else old[c['id']][1] for c in candidate._chunks])
            if vectors.size>1000000:raise ValueError('Dense index exceeds one million stored elements')
        # Commit only after encoding and all budget validation succeed.
        self._documents=candidate._documents;self._chunks=candidate._chunks;self._df=candidate._df;self._vectors=vectors

    def search(self,question,*,k=5,allowed_documents=None):
        np=require('numpy','audio');self._check_encoder()
        # Reuse lexical input/budget validation and filtering even in dense mode.
        if isinstance(allowed_documents,(str,bytes)):raise ValueError('Use a document-ID collection')
        allowed_documents=None if allowed_documents is None else tuple(allowed_documents)
        lexical=super().search(question,k=100,allowed_documents=allowed_documents)
        if type(k) is not int or not 1<=k<=100:raise ValueError('k must be from1 through100')
        if not self._chunks:return []
        query=np.asarray(self.encoder.encode([question]),dtype=np.float32)
        if query.shape!=(1,self._vectors.shape[1]) or not np.isfinite(query).all() or np.linalg.norm(query)==0:raise ValueError('Query encoder shape/value mismatch')
        scores=np.clip(self._vectors@(query[0]/np.linalg.norm(query[0])),-1,1)
        allowed=None if allowed_documents is None else set(allowed_documents)
        indices=[i for i,c in enumerate(self._chunks) if allowed is None or c['document_id'] in allowed]
        ranked=sorted(indices,key=lambda i:(-float(scores[i]),self._chunks[i]['id']))
        dense_ranks={self._chunks[i]['id']:r+1 for r,i in enumerate(ranked) if scores[i]>=self.ranking['min_similarity']}
        lexical_ranks={row['id']:r+1 for r,row in enumerate(lexical)}
        result=[]
        for i in indices:
            chunk=self._chunks[i];key=chunk['id'];dense=dense_ranks.get(key);sparse=lexical_ranks.get(key)
            if dense is None and (self.ranking['mode']=='dense' or sparse is None):continue
            if self.ranking['mode']=='dense':score=float(scores[i])
            else:
                weight=self.ranking['lexical_weight'];offset=self.ranking['rrf_k']
                score=(weight/(offset+sparse) if sparse else 0)+((1-weight)/(offset+dense) if dense else 0)
            result.append(dict({n:v for n,v in chunk.items() if n not in {'terms','length'}},score=score,dense_score=float(scores[i])))
        result=sorted(result,key=lambda r:(-r['score'],r['id']))[:100]
        if self.reranker is not None and result:
            reranked=np.asarray(self.reranker(question,[dict(r) for r in result]),dtype=float)
            if reranked.shape!=(len(result),) or not np.isfinite(reranked).all():raise ValueError('Reranker must return one finite score per candidate')
            result=[dict(row,score=float(score)) for row,score in zip(result,reranked)]
            result.sort(key=lambda r:(-r['score'],r['id']))
        return result[:k]

    def save(self,path):
        self._check_encoder();np=require('numpy','audio')
        vectors=np.empty((0,0),dtype='<f4') if self._vectors is None else self._vectors.astype('<f4')
        raw=vectors.tobytes()
        return _save(path,'rag-semantic-index',dict(config=self.config,ranking=self.ranking,encoder_id=self.encoder_id,
            documents=self._documents,shape=list(vectors.shape),vectors=base64.b64encode(raw).decode(),sha256=hashlib.sha256(raw).hexdigest()))

    @classmethod
    def load(cls,path,*,encoder,reranker=None,reranker_id=None):
        np=require('numpy','audio');record=_load(path,'rag-semantic-index')
        if record['encoder_id']!=getattr(encoder,'fingerprint',None) or record['ranking']['reranker_id']!=reranker_id:raise ValueError('Encoder/reranker identity mismatch')
        ranking=dict(record['ranking']);ranking.pop('reranker_id')
        index=cls(encoder,reranker=reranker,reranker_id=reranker_id,**ranking,**record['config'])
        candidate=Index(**index.config)
        candidate.add_many(dict(id=key,**row) for key,row in record['documents'].items())
        shape=record['shape']
        if not isinstance(shape,list) or len(shape)!=2 or any(type(n) is not int or n<0 for n in shape) or shape[0]!=len(candidate._chunks) or shape[0]*shape[1]>1000000 or shape[0] and not 1<=shape[1]<=4096:raise ValueError('Invalid embedding shape')
        raw=base64.b64decode(record['vectors'],validate=True)
        if len(raw)!=4*shape[0]*shape[1] or hashlib.sha256(raw).hexdigest()!=record['sha256']:raise ValueError('Embedding integrity mismatch')
        vectors=np.frombuffer(raw,dtype='<f4').reshape(shape).copy()
        if not np.isfinite(vectors).all() or len(vectors) and not np.allclose(np.linalg.norm(vectors,axis=1),1,atol=1e-5):raise ValueError('Invalid normalized stored vectors')
        index._documents=candidate._documents;index._chunks=candidate._chunks;index._df=candidate._df;index._vectors=vectors
        return index
