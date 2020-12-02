#!/bin/bash

InDirectory="examples"
OutDirectory="try_cutter_tools"

shopt -s nullglob #nullglob removes null globs from the list of files

for img in "$InDirectory"/*.{jpg,jpeg,png} ;
do
    echo $img
    python3 img_cutter.py --img $img --out_dir $OutDirectory --out_settings "${OutDirectory}/settings.json"
done
