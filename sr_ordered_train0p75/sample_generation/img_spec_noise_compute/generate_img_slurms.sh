#!/bin/bash

# Loop from 0 to 855
for i in $(seq 0 855); do
    # Define the new filename
    new_file="compute_noise_img${i}.slurm"
    
    # Replace the word 'index' in the template with the current value of $i
    sed "s/index/${i}/g" compute_noise_template.slurm > "$new_file"
done
