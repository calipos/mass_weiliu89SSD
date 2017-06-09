cd /home/libailing/nn/self_mask_dssd/weiliu89-caffe-ssd
/home/libailing/nn/self_mask_dssd/weiliu89-caffe-ssd/build/tools/caffe train \
--solver="/home/libailing/nn/self_mask_dssd/weiliu89-caffe-ssd/test/solver_seg.prototxt" \
--gpu 0 2>&1 | tee /home/libailing/nn/self_mask_dssd/weiliu89-caffe-ssd/test/log.log
