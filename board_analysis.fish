#!/usr/bin/env fish

# Loop over layers 0–8
for layer in (seq 0 8)
    # Loop over modes
    for mode in championship random synthetic
        # Loop over mid_dim values
        for mid_dim in 2 4 8 16 32 64 128 256 512
            set data_path probe-data/{$layer}_{$mode}.npz
            set output_prefix board-analysis/layer{$layer}_mode_{$mode}_dim{$mid_dim}
            echo "Running: layer=$layer, mode=$mode, mid_dim=$mid_dim"
            mkdir -p (dirname $output_prefix)
            python board_analysis.py \
                --layer $layer \
                --mode $mode \
                --mid_dim $mid_dim \
                --data_path $data_path \
                --output_prefix $output_prefix
        end
    end
end