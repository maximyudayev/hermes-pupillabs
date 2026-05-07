@echo on
call .venv\Scripts\activate
call hermes-cli -o .\data --config_file .\examples\pupiluvc.yml --experiment project=Test type=PupilUvc trial=0
