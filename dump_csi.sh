#! /bin/bash
#
# run reconstruction batch on a CSI dataset to produce output for tracking testing
#
# Run this script in an appropriate docker container.
#
# > dump_csi.sh /path/to/csi/dataset /path/to/put/dump
#
src_dir=$1
tgt_dir=$2
/build/reconstruction/reconstruction_batch \
    --settings-file /src/config/demo_app.info \
    --csi-dir ${src_dir} \
    -Svolumetric_settings.block_store_path=${tgt_dir} \
    -Straining_img_blocking=true \
    -Straining_min_interval=1 \
    -Straining_max_interval=1
