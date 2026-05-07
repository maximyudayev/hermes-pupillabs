#!/bin/sh
source .venv/bin/activate
hermes-cli -o ./data --config_file ./examples/pupiluvc.yml --experiment project=Test type=PupilUvc trial=0
