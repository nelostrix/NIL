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

Lexical BM25; no learned dense embeddings. Callback citations are validated syntactically, not semantically. Limits:1000 documents,8M characters,10000 chunks by default.

See docs/SEVEN_WORKFLOWS.md and the bundled workflow examples for complete protocols and acceptance.
