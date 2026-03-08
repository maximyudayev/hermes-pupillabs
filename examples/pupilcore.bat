@echo on
call .venv\Scripts\activate
call hermes-cli -o .\data --config_file pupilcore.yml --experiment project=Test type=PupilCore trial=0
