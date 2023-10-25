#!/usr/bin/env bash

tests=$@

if [ -z "$tests" ]; then
    # Run all the tests in this directory.
    # Warning, can take a while to run (> 35 minutes).
    pytest -p no:warnings --verbose
else
    # Run only a user supplied set of tests. For example:
    # $ bash ./runtests.sh test_CH4.py test_H2O.py
    # will run those tests for CH4 and H2O only.
    pytest -p no:warnings --verbose $tests
fi
