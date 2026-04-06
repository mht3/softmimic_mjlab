#!/bin/bash
set -e

sudo apt install -y libyaml-cpp-dev libboost-all-dev libeigen3-dev libspdlog-dev libfmt-dev

pip install -e .
pip install -e algorithms/rsl_rl
