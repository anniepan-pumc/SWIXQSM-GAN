#!/bin/bash

for dir in /Volumes/QSM/shunyi_followup_QSM/*; do
    if [ -d "$dir" ]; then
        base_name=$(basename "$dir")
        mkdir -p /Volumes/QSM/shunyi_followup_dcmmixed/$base_name
        cp -r $dir/*-mixed /Volumes/QSM/shunyi_followup_dcmmixed/$base_name
    else
        echo "Error: $dir is not a directory"
        exit 1
    fi
done