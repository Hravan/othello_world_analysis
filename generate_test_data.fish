#!/usr/bin/env fish

# Loop over layer values 0 through 8
for layer in (seq 0 8)
    # Loop over mode values
    for mode in random championship synthetic
        set output_file probe-data/{$layer}_{$mode}.npz
        echo "Running with layer=$layer mode=$mode -> $output_file"
        mkdir -p probe-data
        python generate_test_data.py --layer $layer --mode $mode --output_path $output_file
    end
end