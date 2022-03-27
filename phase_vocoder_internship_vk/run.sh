#!/bin/bash

pip3 install -r requirements.txt > /dev/null 2>&1
python3 phase_vocoder.py $1 $2 $3
