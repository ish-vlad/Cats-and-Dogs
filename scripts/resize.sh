#!/bin/bash

input_dir=$1
output_dir=$2

mkdir -p $output_dir	

for i in $( ls $input_dir ); do
    convert $input_dir$i -resize 224x224^ $output_dir$i 
    convert $output_dir$i -gravity Center -crop 224x224+0+0 $output_dir$i
done
