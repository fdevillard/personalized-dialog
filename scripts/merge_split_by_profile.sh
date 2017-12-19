#!/bin/bash

read_path='../data/personalized-dialog-dataset/split-by-profile/'
output_path='../data/personalized-dialog-dataset/merged-from-split-by-profile/'
type_files_to_copy='dev trn tst-OOV tst'
base_file_name='personalized-dialog-task5-full-dialogs'
extension='txt'

mkdir -p $output_path

for type in $type_files_to_copy; do
    name_mask="$type.$extension"
    output_file="$base_file_name-$type.$extension"
    find $read_path -name "*-$name_mask" | xargs cat > $output_path/$output_file
done;
