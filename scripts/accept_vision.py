"""Explicit offline image dataset preparation and installed-wheel CPU acceptance."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
import sys
import time


def prepare_digits(destination):
    """Use sklearn's bundled real 8x8 glyphs; no network or implicit install."""
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from PIL import Image
    destination=Path(destination)
    destination.mkdir(parents=True,exist_ok=False)
    digits=load_digits()
    train,valid=train_test_split(list(range(len(digits.target))),train_size=1000,test_size=300,
                                 stratify=digits.target,random_state=41)
    for split,indices in [('train',train),('valid',valid)]:
        for index in indices:
            path=destination/split/str(int(digits.target[index]))/f'{index:04}.png'
            path.parent.mkdir(parents=True,exist_ok=True)
            Image.fromarray((digits.images[index]*255/16).round().astype('uint8')).save(path)
    metadata={'dataset':'sklearn.datasets.load_digits (bundled, no download)',
              'source_version':version('scikit-learn'),'split_seed':41,
              'train_indices':list(map(int,train)),'valid_indices':list(map(int,valid)),
              'pixel_conversion':'round(original_0_to_16 * 255 / 16), uint8 grayscale PNG'}
    (destination/'source.json').write_text(json.dumps(metadata,indent=2)+'\n')


def accept(dataset, output):
    import nil
    from importlib.metadata import distribution
    package=distribution('nil-sdk')
    editable=json.loads(package.read_text('direct_url.json') or '{}').get('dir_info',{}).get('editable',False)
    if editable or Path(nil.__file__).resolve()!=package.locate_file('nil/__init__.py').resolve():
        raise RuntimeError('Acceptance requires a regular installed NIL wheel')
    dataset=Path(dataset).resolve();output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1]
    started=time.monotonic()
    with (output/'execution.log').open('wb') as log:
        completed=subprocess.run([sys.executable,'-I',str(root/'nil-sdk/examples/image_classification.py'),
            '--train',str(dataset/'train'),'--valid',str(dataset/'valid'),'--epochs','60','--size','16',
            '--width','16','--output',str(output/'workflow')],stdout=log,stderr=subprocess.STDOUT)
    report_path=output/'workflow/report.json'
    report=json.loads(report_path.read_text()) if report_path.exists() else {}
    passed=completed.returncode==0 and report.get('after',{}).get('accuracy',0)>=.80
    evidence={'schema_version':1,'status':'passed' if passed else 'failed','returncode':completed.returncode,
        'scope':'Installed-wheel CPU image classification on 1000 train/300 held-out bundled digit glyphs; not general vision accuracy',
        'minimum_held_out_accuracy':.80,'workflow':report,
        'python':platform.python_version(),'platform':platform.platform(),
        'versions':{name:version(name) for name in ('nil-sdk','torch','numpy','Pillow')},
        'seconds':time.monotonic()-started,'hardware_verified':False,
        'dataset_source':json.loads((dataset/'source.json').read_text())}
    try:
        import resource
        raw=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        evidence['peak_child_rss_bytes']=raw if sys.platform=='darwin' else raw*1024
    except ImportError:
        evidence['peak_child_rss_bytes']=None
    (output/'acceptance.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps({k:v for k,v in evidence.items() if k not in {'dataset_source','workflow'}},indent=2))
    return 0 if passed else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare-digits');prep.add_argument('--output',type=Path,required=True)
    check=sub.add_parser('check');check.add_argument('--dataset',type=Path,required=True);check.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='prepare-digits':prepare_digits(args.output);return 0
    return accept(args.dataset,args.output)

if __name__=='__main__':raise SystemExit(main())
