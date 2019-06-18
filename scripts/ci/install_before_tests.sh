#!/usr/bin/env bash

# from https://github.com/justusschock/template-repo-python/blob/master/scripts/ci/install_before_tests.sh
pip install -U pip wheel;
pip install -r requirements.txt;
pip install coverage;
pip install codecov;