"""Bounded lexical retrieval, provenance and extractive or explicit-generator RAG."""
from collections import Counter
import hashlib
import math
import re
from nil._workflow_io import save as _save, load as _load


def _tokens(text):
    return re.findall(r'\w+', text.casefold())


class Index:
    def __init__(self, *, chunk_chars=1000, overlap=100, max_documents=1000, max_chars=8000000):
        for name,value in [('chunk_chars',chunk_chars),('max_documents',max_documents),('max_chars',max_chars)]:
            if type(value) is not int or not 1 <= value <= 8000000:
                raise ValueError(f'Invalid {name}')
        if type(overlap) is not int or not 0 <= overlap < chunk_chars:
            raise ValueError('overlap must be smaller than chunk_chars')
        self.config=dict(chunk_chars=chunk_chars,overlap=overlap,max_documents=max_documents,max_chars=max_chars)
        self._documents={};self._chunks=[];self._df=Counter()

    def _rebuild(self, documents):
        chunks=[];df=Counter();size=self.config['chunk_chars'];stride=size-self.config['overlap']
        for key,document in sorted(documents.items()):
            for start in range(0,len(document['text']),stride):
                text=document['text'][start:start+size];terms=Counter(_tokens(text))
                chunks.append(dict(id=f'{key}:{start}',document_id=key,source=document['source'],
                                   start=start,end=start+len(text),text=text,
                                   document_sha256=hashlib.sha256(document['text'].encode()).hexdigest(),terms=terms,length=sum(terms.values())))
                df.update(terms.keys())
                if len(chunks)>10000:raise ValueError('Index exceeds 10000 chunks')
                if start+size>=len(document['text']):break
        self._documents=documents;self._chunks=chunks;self._df=df

    def add(self, document_id, text, *, source=None):
        if not isinstance(document_id,str) or not document_id or len(document_id)>128 or ':' in document_id:
            raise ValueError('document_id requires 1–128 characters without a colon')
        if not isinstance(text,str) or not text.strip():raise ValueError('Document text must be nonempty')
        source=document_id if source is None else source
        if not isinstance(source,str) or len(source)>2048:raise ValueError('Invalid source')
        candidate={**self._documents,document_id:dict(text=text,source=source)}
        if len(candidate)>self.config['max_documents'] or sum(len(v['text']) for v in candidate.values())>self.config['max_chars']:
            raise ValueError('Index document/character budget exceeded')
        self._rebuild(candidate);return self

    def delete(self, document_id):
        if document_id not in self._documents:raise KeyError(document_id)
        self._rebuild({k:v for k,v in self._documents.items() if k!=document_id})

    def search(self, question, *, k=5, allowed_documents=None):
        if not isinstance(question,str) or not question.strip() or len(question)>10000:raise ValueError('Invalid question')
        if type(k) is not int or not 1<=k<=100:raise ValueError('k must be from 1 through 100')
        if allowed_documents is not None:
            if isinstance(allowed_documents,(str,bytes)):raise ValueError('Use a document-ID collection')
            allowed_documents=set(allowed_documents)
        query=set(_tokens(question));count=len(self._chunks)
        if not count:return []
        average=sum(c['length'] for c in self._chunks)/count or 1
        scored=[]
        for chunk in self._chunks:
            if allowed_documents is not None and chunk['document_id'] not in allowed_documents:continue
            score=0.
            for token in query:
                freq=chunk['terms'][token]
                if freq:
                    idf=math.log(1+(count-self._df[token]+.5)/(self._df[token]+.5))
                    score+=idf*freq*2.2/(freq+1.2*(.25+.75*chunk['length']/average))
            if score>0:scored.append(dict({key:value for key,value in chunk.items() if key not in {'terms','length'}},score=score))
        return sorted(scored,key=lambda row:(-row['score'],row['id']))[:k]

    def ask(self, question, *, k=3, generator=None, allowed_documents=None):
        hits=self.search(question,k=k,allowed_documents=allowed_documents)
        if not hits:return dict(text='',citations=[],abstained=True)
        if generator is None:answer=dict(text='\n'.join(row['text'] for row in hits),citations=[row['id'] for row in hits])
        else:answer=generator(question,[dict(row) for row in hits])
        if not isinstance(answer,dict) or set(answer)!={'text','citations'} or not isinstance(answer['text'],str) or not answer['text'].strip() or len(answer['text'])>100000 or not isinstance(answer['citations'],list) or not answer['citations'] or any(not isinstance(c,str) or c not in {r['id'] for r in hits} for c in answer['citations']):
            raise ValueError('Generator must return bounded text and citations to supplied chunks')
        return dict(answer,abstained=False)

    def save(self,path):return _save(path,'rag-index',dict(config=self.config,documents=self._documents))

    @classmethod
    def load(cls,path):
        record=_load(path,'rag-index');index=cls(**record['config'])
        for key,row in record['documents'].items():index.add(key,row['text'],source=row['source'])
        return index


def evaluate(index, queries, *, k=5):
    recalls=[];ranks=[]
    for i,row in enumerate(queries):
        if i>=10000:raise ValueError('Evaluation exceeds 10000 queries')
        expected=set(row['relevant_documents'])
        if not expected:raise ValueError('Each query needs relevant document IDs')
        hits=index.search(row['question'],k=k)
        documents=list(dict.fromkeys(h['document_id'] for h in hits))
        recalls.append(len(expected & set(documents))/len(expected))
        ranks.append(next((1/(n+1) for n,d in enumerate(documents) if d in expected),0.))
    if not recalls:raise ValueError('Empty evaluation')
    return {'queries':len(recalls),f'recall@{k}':sum(recalls)/len(recalls),'mrr':sum(ranks)/len(ranks)}
