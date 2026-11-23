"""Portable EPERM policy tests; real macOS acceptance is a separate CI gate."""
from types import SimpleNamespace
import signal
import pytest
from nil import execution


def test_darwin_zombie_only_group_does_not_turn_cleanup_into_failure(monkeypatch):
    calls=[]
    def refuse(*args):calls.append(args);raise PermissionError('Darwin zombie group')
    monkeypatch.setattr(execution.os,'killpg',refuse)
    monkeypatch.setattr(execution,'sys',SimpleNamespace(platform='darwin'))
    monkeypatch.setattr(execution.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=0,stdout='123 Z\n456 S\n'))
    execution._signal_group(123,signal.SIGTERM)
    assert calls==[(123,signal.SIGTERM)]


@pytest.mark.parametrize('output',['123 S\n','malformed','123 U\n'])
def test_live_or_unknown_group_permission_failure_is_not_suppressed(monkeypatch,output):
    def refuse(*args):raise PermissionError('real ownership failure')
    monkeypatch.setattr(execution.os,'killpg',refuse)
    monkeypatch.setattr(execution,'sys',SimpleNamespace(platform='darwin'))
    monkeypatch.setattr(execution.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=0,stdout=output))
    with pytest.raises(PermissionError):execution._signal_group(123,signal.SIGTERM)
