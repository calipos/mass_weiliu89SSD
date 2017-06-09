cd /home/libailing/nn/self_mask_dssd/weiliu89-caffe-ssd
/home/libailing/nn/self_mask_dssd/weiliu89-caffe-ssd/build/tools/caffe train \
--solver="/home/libailing/nn/self_mask_dssd/weiliu89-caffe-ssd/test/solver_ssd.prototxt" \
--gpu 0 2>&1 | tee /home/libailing/data/fujian_QuanZhou_ctpt_yaban/jobs/fujian_QuanZhou_ctpt_yaban_SSD_400x300.log
