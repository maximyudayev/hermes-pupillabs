#!/bin/sh
source .venv/bin/activate
hermes-cli -o ./data --config_file ./examples/pupilcore.yml --experiment project=Test type=PupilCore trial=0
