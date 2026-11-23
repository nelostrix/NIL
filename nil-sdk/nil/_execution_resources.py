"""Explicit Linux resource controls; no filesystem writes or threads at import."""
import os
from pathlib import Path
import re
import sys
import time


class ResourceUnavailable(RuntimeError):
    """A requested resource guarantee cannot be established on this host."""


def _mount(path):
    """Resolve the closest mount, including bind mounts and escaped mount names."""
    matches=[]
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        left,right=line.split(' - ',1)
        point=Path(re.sub(r'\\([0-7]{3})',lambda m:chr(int(m[1],8)),left.split()[4]))
        if path.is_relative_to(point): matches.append((len(point.parts),right.split()[0]))
    return max(matches,default=(0,None))[1]


def validate_cgroup_root(root):
    if sys.platform!='linux': raise ResourceUnavailable('cgroup-v2 memory containment requires Linux')
    path=Path(root)
    if not path.is_absolute() or path.resolve()!=path or not path.is_dir():
        raise ResourceUnavailable('cgroup_root must be an existing absolute directory without symlinks')
    try:
        if _mount(path)!='cgroup2': raise ResourceUnavailable('cgroup_root is not on a cgroup-v2 mount')
        enabled=(path/'cgroup.subtree_control').read_text().split()
        if 'memory' not in enabled: raise ResourceUnavailable('Delegate and enable the memory controller before requesting aggregate containment')
        if not os.access(path,os.W_OK|os.X_OK) or not os.access(path/'cgroup.procs',os.W_OK):
            raise ResourceUnavailable('cgroup_root is not delegated/writable by the current user')
    except OSError as error: raise ResourceUnavailable(f'Cannot inspect delegated cgroup: {error.strerror}') from error
    return path


def cgroup_capabilities(root=None):
    result={'configured':root is not None,'available':False,'reason':'No delegated cgroup_root configured'}
    if root is not None:
        try: validate_cgroup_root(root)
        except ResourceUnavailable as error: result['reason']=str(error)
        else: result.update(available=True,reason='Preflight passed; child creation/attachment is checked at job startup')
    return result


def validate_limits(limits):
    if limits.group_rss_mb is not None:
        if sys.platform!='linux' or not Path('/proc/self/stat').is_file():
            raise ResourceUnavailable('Sampled process-group RSS enforcement requires readable Linux procfs')
        group_rss(os.getpgrp())  # Refuse unreadable existing group members before launch.
    if limits.aggregate_memory_mb is not None: validate_cgroup_root(limits.cgroup_root)


def group_rss(group,proc=Path('/proc')):
    """Sum member RSS (shared pages count per process), not unique physical RAM."""
    pages=0; members=0
    try: entries=list(proc.iterdir())
    except OSError as error: raise ResourceUnavailable('Cannot enumerate Linux process RSS') from error
    for entry in entries:
        if not entry.name.isdigit(): continue
        try: fields=(entry/'stat').read_text().rpartition(') ')[2].split()
        except (FileNotFoundError,ProcessLookupError): continue
        except PermissionError as error:
            # Restricted procfs cannot establish a complete aggregate measurement.
            raise ResourceUnavailable('Process RSS is hidden by procfs permissions') from error
        try:
            if int(fields[2])==group and fields[0]!='Z':
                pages+=max(0,int(fields[21])); members+=1
        except (IndexError,ValueError) as error: raise ResourceUnavailable('Malformed Linux process accounting') from error
    return {'rss_bytes':pages*os.sysconf('SC_PAGE_SIZE'),'processes':members}


class Cgroup:
    """A single explicitly delegated child; kernel controls are not emulated."""
    def __init__(self,root,job_id,memory_mb):
        if not re.fullmatch(r'job-[0-9a-f]{32}',job_id): raise ValueError('Invalid owned cgroup name')
        self.path=validate_cgroup_root(root)/('nil-'+job_id)
        self.memory_bytes=memory_mb*1024**2
        self.path.mkdir()
        try:
            for name,value in (('memory.max',self.memory_bytes),('memory.swap.max',0),('memory.oom.group',1)):
                control=self.path/name
                with control.open('r+') as stream:
                    stream.write(str(value)); stream.flush(); stream.seek(0)
                    if stream.read().strip()!=str(value): raise ResourceUnavailable(f'Kernel did not apply {name}')
            if not os.access(self.path/'cgroup.kill',os.W_OK): raise ResourceUnavailable('Whole-cgroup cleanup requires writable cgroup.kill')
        except BaseException:
            self.path.rmdir(); raise
    def evidence(self):
        events={key:int(value) for key,value in (line.split() for line in (self.path/'memory.events').read_text().splitlines())}
        peak=self.path/'memory.peak'
        return {'backend':'cgroup-v2','memory_limit_bytes':self.memory_bytes,'swap_limit_bytes':0,
                'memory_peak_bytes':int(peak.read_text()) if peak.exists() else None,'memory_events':events}
    def kill(self):
        with (self.path/'cgroup.kill').open('w') as stream: stream.write('1')
    def close(self):
        deadline=time.monotonic()+2
        while True:
            try:
                # Remove only owned descendants after cgroup.kill, deepest first.
                for current,children,files in os.walk(self.path,topdown=False):
                    Path(current).rmdir()
                if self.path.exists(): self.path.rmdir()
                return
            except OSError:
                if time.monotonic()>=deadline: raise
                time.sleep(.02)


def attach(path):
    """Called before input deserialization, so no user operation runs uncontained."""
    with (Path(path)/'cgroup.procs').open('r+') as stream: stream.write(str(os.getpid()))
