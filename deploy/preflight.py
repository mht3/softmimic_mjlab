"""Read-only clone/network preflight. Never initializes DDS or publishes commands."""
import argparse
import json
from pathlib import Path
import socket
import subprocess
from check_bundle import ROOT, check


def check_network(name, sysnet=Path('/sys/class/net')):
    if name == 'lo' or name not in dict((n, i) for i,n in socket.if_nameindex()):
        raise ValueError('Select an existing robot Ethernet interface, not loopback')
    path=sysnet/name
    if (path/'wireless').exists() or (path/'type').read_text().strip() != '1':
        raise ValueError('The selected adapter is not an Ethernet interface')
    if (path/'operstate').read_text().strip() != 'up' or (path/'carrier').read_text().strip() != '1':
        raise ValueError('Ethernet link is not UP with carrier; check cable and adapter configuration')
    info=json.loads(subprocess.check_output(['ip','-j','-4','addr','show','dev',name],text=True))
    addresses=[f"{a['local']}/{a['prefixlen']}" for i in info for a in i['addr_info'] if a['family']=='inet']
    if not addresses: raise ValueError('Ethernet adapter has no IPv4 address; configure the robot subnet')
    print(f'PASS: Ethernet {name}, carrier up, IPv4 {", ".join(addresses)}. Confirm this matches the robot subnet.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--network',help='Check an Ethernet adapter locally; does not contact the robot')
    a=p.parse_args()
    project=ROOT/'deploy/robots/g1_23dof'
    check(project/'config/policy/compliant_mimic/stand',project/'config/config.yaml')
    binary=project/'build/g1_ctrl'
    if not binary.is_file(): raise ValueError('Build g1_ctrl first using the commands in deploy/README.md')
    # --help exits before ChannelFactory initialization in main.cpp.
    result=subprocess.run([str(binary),'--help'],capture_output=True,text=True,timeout=10)
    if result.returncode or '--network' not in result.stdout:
        raise ValueError('Controller executable/dependencies failed: '+result.stderr)
    print('PASS: native executable loads and --help exits without opening DDS.')
    if a.network: check_network(a.network)
    else: print('Network not checked. Rerun with --network=YOUR_ETHERNET_INTERFACE on the deployment PC.')
    print('No hardware contacted. Firmware/mode, actual sensor timing and physical behavior require a robot-side trial.')

if __name__=='__main__':main()
