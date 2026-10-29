from pathlib import Path
import json, random, platform
import numpy as np

def find_root(p=None):
 p=Path(p or Path.cwd()).resolve()
 for x in [p,*p.parents]:
  if (x/'src').is_dir() and (x/'data').is_dir(): return x
 raise FileNotFoundError('Project root with src/ and data/ not found')

def setup(p=None):
 random.seed(42); np.random.seed(42); r=find_root(p)
 for d in ['metrics','figures','tables']: (r/'results'/d).mkdir(exist_ok=True)
 return r

def save(path,obj):
 Path(path).write_text(json.dumps(obj,indent=2,default=lambda x: x.item() if hasattr(x,'item') else str(x)),encoding='utf8')
 
def meta(root,module,**kw): return {'module':module,'python':platform.python_version(),'root':str(root),**kw}
