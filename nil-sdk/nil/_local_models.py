"""Shared explicit local-only model identity and exclusive artifact publication."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile


def identity(path, *, max_bytes=1024**3):
    root=Path(path).resolve()
    if not root.is_dir():raise ValueError('Provide an existing local model directory; download explicitly outside NIL first')
    if type(max_bytes) is not int or not 1<=max_bytes<=16*1024**3:raise ValueError('Invalid model byte budget')
    files=[];total=0
    for entry in sorted(root.iterdir()):
        if entry.is_file() and not entry.name.startswith('.') and entry.suffix in {'.json','.txt','.safetensors','.bin','.model'}:
            total+=entry.stat().st_size
            if total>max_bytes or len(files)>=128:raise ValueError('Local model exceeds file/byte budget')
            digest=hashlib.sha256()
            with entry.open('rb') as stream:
                for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
            files.append([entry.name,digest.hexdigest()])
    if not files or not (root/'config.json').is_file():raise ValueError('Local model requires config.json and model files')
    fingerprint=hashlib.sha256(json.dumps(files,separators=(',',':')).encode()).hexdigest()
    manifest=root/'.nil-model.json'
    if manifest.exists():
        if manifest.stat().st_size>65536:raise ValueError('Model manifest exceeds 64 KiB')
        record=json.loads(manifest.read_text())
        if record.get('schema_version')!=1 or record.get('files')!=files:raise ValueError('Local model integrity check failed')
    return root,fingerprint,files


def publish(model, processor, destination, *, kind):
    root=Path(destination).absolute()
    if root.exists() or root.is_symlink():raise FileExistsError(root)
    root.parent.mkdir(parents=True,exist_ok=True)
    temporary=Path(tempfile.mkdtemp(prefix='.nil-local-model-',dir=root.parent));claimed=False
    try:
        model.save_pretrained(temporary,safe_serialization=True)
        processor.save_pretrained(temporary)
        _,fingerprint,files=identity(temporary,max_bytes=16*1024**3)
        record=dict(schema_version=1,kind=kind,files=files,fingerprint=fingerprint)
        root.mkdir();claimed=True
        for entry in temporary.iterdir():shutil.move(str(entry),root/entry.name)
        with (root/'.nil-model.json').open('x',encoding='utf-8') as stream:
            json.dump(record,stream,indent=2);stream.flush();os.fsync(stream.fileno())
        return root
    except BaseException:
        if claimed:shutil.rmtree(root)
        raise
    finally:shutil.rmtree(temporary)
