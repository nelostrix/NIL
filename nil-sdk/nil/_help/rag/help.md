# rag

## Usage

```python
import nil
index = nil.rag.Index().add('manual', 'Pump temperature is 40 degrees.', source='manual.txt')
print(index.ask('pump temperature'))
```

## Parameters

Index chunk_chars/overlap/document and character budgets; add/delete; search/ask k and allowed_documents; explicit generator callback.

## Output

Ranked chunks with source/offset/hash, cited extractive answer or callback output, abstention, recall/MRR; exclusive index snapshots.

## Requirements

Standard library only.

## Limitations

Base Index uses lexical BM25. Callback citations are validated syntactically, not semantically. Limits:1000 documents,8M characters,10000 chunks by default.

See docs/SEVEN_WORKFLOWS.md for v1 acceptance and docs/ADVANCED_WORKFLOWS.md for this phase and remaining work.

## Optional semantic API

Use load_encoder(existing_local_directory), semantic_index(encoder, mode="hybrid"), add_many(rows), and evaluate_answers(...). Dense embeddings need NumPy and the native local encoder needs explicit Transformers/Torch. Encoder/reranker identities must match on reload. Base Index remains stdlib BM25.
