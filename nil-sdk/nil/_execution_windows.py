"""Windows Job Object ownership using Python's standard library only.

No DLL, handle, thread or process is opened at import. CPython _winapi supplies
pipe/process handles; ctypes supplies extended process creation with a creation-
time job list and the documented Job Object calls. Actual Windows acceptance is a separate gate.
"""
import os
import subprocess
import time


class _Runtime:
    def __init__(self):
        if os.name!='nt': raise NotImplementedError('Windows Job Objects require Windows')
        import ctypes as C
        from ctypes import wintypes as W
        import _winapi
        self.C,self.W,self.win=C,W,_winapi
        kernel=C.WinDLL('kernel32',use_last_error=True)
        class Basic(C.Structure):
            _fields_=[('user',C.c_longlong),('kernel',C.c_longlong),('flags',W.DWORD),
                ('min_working_set',C.c_size_t),('max_working_set',C.c_size_t),
                ('process_limit',W.DWORD),('affinity',C.c_size_t),('priority',W.DWORD),('scheduling',W.DWORD)]
        class IO(C.Structure):
            _fields_=[(name,C.c_ulonglong) for name in ('read_ops','write_ops','other_ops','read_bytes','write_bytes','other_bytes')]
        class Extended(C.Structure):
            _fields_=[('basic',Basic),('io',IO),('process_memory',C.c_size_t),('job_memory',C.c_size_t),
                ('peak_process_memory',C.c_size_t),('peak_job_memory',C.c_size_t)]
        class Accounting(C.Structure):
            _fields_=[('user',C.c_longlong),('kernel',C.c_longlong),('period_user',C.c_longlong),('period_kernel',C.c_longlong),
                ('page_faults',W.DWORD),('total_processes',W.DWORD),('active_processes',W.DWORD),('terminated_processes',W.DWORD)]
        class Startup(C.Structure):
            _fields_=[('cb',W.DWORD),('reserved',W.LPWSTR),('desktop',W.LPWSTR),('title',W.LPWSTR),
                *[(name,W.DWORD) for name in ('x','y','x_size','y_size','x_chars','y_chars','fill','flags')],
                ('show',W.WORD),('reserved_bytes',W.WORD),('reserved_data',C.c_void_p),
                ('stdin',W.HANDLE),('stdout',W.HANDLE),('stderr',W.HANDLE)]
        class StartupEx(C.Structure):
            _fields_=[('basic',Startup),('attributes',C.c_void_p)]
        class Information(C.Structure):
            _fields_=[('process',W.HANDLE),('thread',W.HANDLE),('pid',W.DWORD),('tid',W.DWORD)]
        self.StartupEx,self.Information=StartupEx,Information
        self.Extended,self.Accounting=Extended,Accounting
        def bind(name,args,result):
            method=getattr(kernel,name);method.argtypes=args;method.restype=result;return method
        self.CreateJob=bind('CreateJobObjectW',[C.c_void_p,W.LPCWSTR],W.HANDLE)
        self.SetJob=bind('SetInformationJobObject',[W.HANDLE,C.c_int,C.c_void_p,W.DWORD],W.BOOL)
        self.InJob=bind('IsProcessInJob',[W.HANDLE,W.HANDLE,C.POINTER(W.BOOL)],W.BOOL)
        self.InitAttributes=bind('InitializeProcThreadAttributeList',[C.c_void_p,W.DWORD,W.DWORD,C.POINTER(C.c_size_t)],W.BOOL)
        self.UpdateAttribute=bind('UpdateProcThreadAttribute',[C.c_void_p,W.DWORD,C.c_size_t,C.c_void_p,C.c_size_t,C.c_void_p,C.c_void_p],W.BOOL)
        self.DeleteAttributes=bind('DeleteProcThreadAttributeList',[C.c_void_p],None)
        self.CreateProcess=bind('CreateProcessW',[W.LPCWSTR,W.LPWSTR,C.c_void_p,C.c_void_p,W.BOOL,W.DWORD,C.c_void_p,W.LPCWSTR,C.c_void_p,C.c_void_p],W.BOOL)
        self.Resume=bind('ResumeThread',[W.HANDLE],W.DWORD)
        self.KillJob=bind('TerminateJobObject',[W.HANDLE,W.UINT],W.BOOL)
        self.QueryJob=bind('QueryInformationJobObject',[W.HANDLE,C.c_int,C.c_void_p,W.DWORD,C.POINTER(W.DWORD)],W.BOOL)
        self.OpenProcess=bind('OpenProcess',[W.DWORD,W.BOOL,W.DWORD],W.HANDLE)
        self.ProcessTimes=bind('GetProcessTimes',[W.HANDLE,*([C.POINTER(W.FILETIME)]*4)],W.BOOL)
    def _check(self,result):
        if not result: raise self.C.WinError(self.C.get_last_error())
        return result
    def close(self,handle): self.win.CloseHandle(handle)
    def create_job(self):
        handle=self._check(self.CreateJob(None,None))
        try:
            limits=self.Extended();limits.basic.flags=0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE; no breakaway.
            self._check(self.SetJob(handle,9,self.C.byref(limits),self.C.sizeof(limits)))
            return handle
        except BaseException:self.close(handle);raise
    def create_pipe(self):
        read,write=self.win.CreatePipe(None,0)
        try: os.set_handle_inheritable(write,True)
        except BaseException:self.close(read);self.close(write);raise
        return read,write
    def create_input(self):
        handle=self.win.CreateFile('NUL',0x80000000,3,0,3,0x80,0)
        try:os.set_handle_inheritable(handle,True)
        except BaseException:self.close(handle);raise
        return handle
    def create_suspended(self,args,write,stdin,job):
        C,W=self.C,self.W;size=C.c_size_t()
        self.InitAttributes(None,2,0,C.byref(size))
        if not size.value:raise C.WinError(C.get_last_error())
        attributes=C.create_string_buffer(size.value)
        self._check(self.InitAttributes(attributes,2,0,C.byref(size)))
        try:
            handles=(W.HANDLE*2)(write,stdin);jobs=(W.HANDLE*1)(job)
            self._check(self.UpdateAttribute(attributes,0,0x00020002,handles,C.sizeof(handles),None,None))
            self._check(self.UpdateAttribute(attributes,0,0x0002000D,jobs,C.sizeof(jobs),None,None))
            # JOB_LIST assigns ownership during creation, eliminating the separate
            # CreateProcess/AssignProcess orphan window if the supervisor dies.
            startup=self.StartupEx();startup.basic.cb=C.sizeof(startup);startup.basic.flags=0x100
            startup.basic.stdin=stdin;startup.basic.stdout=write;startup.basic.stderr=write
            startup.attributes=C.cast(attributes,C.c_void_p)
            info=self.Information();command=C.create_unicode_buffer(subprocess.list2cmdline(args))
            self._check(self.CreateProcess(args[0],command,None,None,True,
                0x4|0x08000000|0x00080000|0x00000400,None,None,C.byref(startup),C.byref(info)))
            return info.process,info.thread,int(info.pid),int(info.tid)
        finally:self.DeleteAttributes(attributes)
    def assign(self,job,process):
        # Ownership is already established by JOB_LIST. Verify before resume.
        member=self.W.BOOL();self._check(self.InJob(process,job,self.C.byref(member)))
        if not member.value:raise OSError('Worker was not assigned to its Windows job')
    def resume(self,thread):
        if self.Resume(thread)==0xffffffff:raise self.C.WinError(self.C.get_last_error())
    def terminate(self,process):self.win.TerminateProcess(process,1)
    def kill_job(self,job):self._check(self.KillJob(job,1))
    def active(self,job):
        info=self.Accounting()
        self._check(self.QueryJob(job,1,self.C.byref(info),self.C.sizeof(info),None))
        return int(info.active_processes)
    def poll(self,handle):
        state=self.win.WaitForSingleObject(handle,0)
        if state==258:return None
        if state!=0:raise OSError('Unexpected Windows process wait result')
        return self.win.GetExitCodeProcess(handle)
    def wait(self,handle,timeout=None):
        milliseconds=0xffffffff if timeout is None else min(0xfffffffe,max(0,int(timeout*1000+.999)))
        if self.win.WaitForSingleObject(handle,milliseconds)==258:raise subprocess.TimeoutExpired('owned Windows process',timeout)
        result=self.poll(handle)
        if result is None:raise OSError('Owned Windows process did not exit')
        return result
    def stdout(self,handle):
        import msvcrt
        try:fd=msvcrt.open_osfhandle(handle,os.O_RDONLY|os.O_BINARY)
        except BaseException:self.close(handle);raise
        try:return os.fdopen(fd,'rb',buffering=0)
        except BaseException:os.close(fd);raise
    def identity(self,pid):
        if type(pid) is not int or not 0<pid<=0xffffffff:return None
        handle=self.OpenProcess(0x100000|0x1000,False,pid)  # SYNCHRONIZE | QUERY_LIMITED_INFORMATION
        if not handle:
            error=self.C.get_last_error()
            if error==87:return None  # No such process.
            raise self.C.WinError(error)
        try:
            if self.poll(handle) is not None:return None
            times=[self.W.FILETIME() for _ in range(4)]
            self._check(self.ProcessTimes(handle,*[self.C.byref(value) for value in times]))
            return str(times[0].dwLowDateTime|(times[0].dwHighDateTime<<32))
        finally:self.close(handle)


class _Process:
    def __init__(self,runtime,handle,pid,args,stdout):
        self.runtime,self.handle,self.pid,self.args,self.stdout=runtime,handle,pid,args,stdout
        self.returncode=None
    def poll(self):
        if self.returncode is None:self.returncode=self.runtime.poll(self.handle)
        return self.returncode
    def wait(self,timeout=None):
        if self.returncode is None:self.returncode=self.runtime.wait(self.handle,timeout)
        return self.returncode
    def close(self):
        if self.handle is not None:self.runtime.close(self.handle);self.handle=None


class _Tree:
    def __init__(self,runtime,handle,process):self.runtime,self.handle,self.process=runtime,handle,process
    def stop(self):
        if self.handle is None:return
        self.runtime.kill_job(self.handle)
        self.process.wait(timeout=5)
        deadline=time.monotonic()+5
        while self.runtime.active(self.handle):
            if time.monotonic()>=deadline:raise TimeoutError('Windows job descendants did not exit')
            time.sleep(.01)
    def close(self):
        try:
            if self.handle is not None:self.runtime.close(self.handle);self.handle=None
        finally:self.process.close()


def launch(args):
    runtime=_Runtime();job=read=write=stdin=process=thread=None;stdout=None
    try:
        job=runtime.create_job();read,write=runtime.create_pipe();stdin=runtime.create_input()
        process,thread,pid,_=runtime.create_suspended(args,write,stdin,job)
        runtime.assign(job,process)  # No user instruction may run before ownership succeeds.
        transferred,read=read,None;stdout=runtime.stdout(transferred)
        worker=_Process(runtime,process,pid,args,stdout);tree=_Tree(runtime,job,worker)
        runtime.resume(thread)
        runtime.close(thread);thread=None
        runtime.close(write);write=None
        runtime.close(stdin);stdin=None
        return worker,tree
    except BaseException:
        # Every error path closes the job and kills even an unassigned suspended process.
        if process is not None:
            try:runtime.terminate(process);runtime.wait(process,5)
            except (OSError,subprocess.TimeoutExpired):pass
        if stdout is not None:
            try:stdout.close()
            except OSError:pass
        for handle in (process,job):
            if handle is not None:
                try:runtime.close(handle)
                except OSError:pass
        raise
    finally:
        for handle in (thread,read,write,stdin):
            if handle is not None:
                try:runtime.close(handle)
                except OSError:pass


def identity(pid):return _Runtime().identity(pid)
