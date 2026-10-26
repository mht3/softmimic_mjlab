"""Preflight fails closed and console lifecycle never needs a robot."""
from pathlib import Path
import json
import os
import pty
import subprocess
import sys
import termios
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'deploy'))
import preflight


def test_local_network_preflight(tmp_path,monkeypatch,capsys):
    monkeypatch.setattr(preflight.socket,'if_nameindex',lambda:[(1,'lo'),(2,'robot_eth')])
    dev=tmp_path/'robot_eth';dev.mkdir()
    for name,value in {'type':'1','operstate':'up','carrier':'1'}.items():(dev/name).write_text(value)
    monkeypatch.setattr(preflight.subprocess,'check_output',lambda *args,**kwargs:json.dumps([{'addr_info':[{'family':'inet','local':'192.168.123.222','prefixlen':24}]}]))
    preflight.check_network('robot_eth',tmp_path)
    assert '192.168.123.222/24' in capsys.readouterr().out
    with pytest.raises(ValueError,match='existing'):preflight.check_network('lo',tmp_path)
    (dev/'carrier').write_text('0')
    with pytest.raises(ValueError,match='carrier'):preflight.check_network('robot_eth',tmp_path)


def test_keyboard_shutdown_headless_and_partial_escape(tmp_path):
    source=tmp_path/'keyboard.cpp';exe=tmp_path/'keyboard'
    source.write_text('#include "isaaclab/devices/keyboard/keyboard.h"\nint main(){Keyboard k;std::this_thread::sleep_for(std::chrono::milliseconds(150));k.update();k.key();}\n')
    subprocess.run(['g++','-std=c++17','-pthread','-I'+str(ROOT/'deploy/include'),str(source),'-o',str(exe)],check=True)
    subprocess.run([str(exe)],stdin=subprocess.DEVNULL,check=True,timeout=3)
    master,slave=pty.openpty()
    before=termios.tcgetattr(slave)
    try:
        process=subprocess.Popen([str(exe)],stdin=slave,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
        os.write(master,b'\x1b')
        _,stderr=process.communicate(timeout=3)
        assert process.returncode==0,stderr
        assert termios.tcgetattr(slave)==before
    finally:
        os.close(master);os.close(slave)
