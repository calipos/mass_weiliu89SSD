#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <algorithm>
#include <map>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/segment_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/sampler.hpp"

namespace caffe {

template <typename Dtype>
SegmentedDataLayer<Dtype>::SegmentedDataLayer(const LayerParameter& param)
  : BasePrefetchingDataLayer<Dtype>(param),
    reader_(param) {
}

template <typename Dtype>
SegmentedDataLayer<Dtype>::~SegmentedDataLayer() {
  this->StopInternalThread();
}

template <typename Dtype>
void SegmentedDataLayer<Dtype>::DataLayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int batch_size = this->layer_param_.data_param().batch_size();

  // Read a data point, and use it to initialize the top blob.
  SegmentDatum& seg_datum = *(reader_.full().peek());
  // Use data_transformer to infer the expected blob shape from anno_datum.
  vector<int> top_img_shape = this->data_transformer_->InferBlobShape(seg_datum.imgdata()); 
  this->transformed_img_data_.Reshape(top_img_shape);
  top_img_shape[0] = batch_size;

  top[0]->Reshape(top_img_shape);
  	for (int i = 0; i < this->PREFETCH_COUNT; ++i)
	   {
	    this->prefetch_[i].data_.Reshape(top_img_shape);
	  }
  LOG(INFO) << "output data size: " << top[0]->num() << ","<< top[0]->channels() << "," << top[0]->height() << ","<< top[0]->width();

    vector<int> top_seg_shape =this->data_transformer_->InferBlobShape(seg_datum.maskdata());
    this->transformed_seg_data_.Reshape(top_seg_shape);
    top_seg_shape[0] = batch_size;
    top[1]->Reshape(top_seg_shape);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].label_.Reshape(top_seg_shape);
    }
    LOG(INFO) << "output label size: " << top[1]->num() << ","<< top[1]->channels() << "," << top[1]->height() << ","<< top[1]->width();
  
}

// This function is called on prefetch thread
template<typename Dtype>
void SegmentedDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_img_data_.count());
  CHECK(this->transformed_seg_data_.count());

  // Reshape according to the first anno_datum of each batch
  // on single input batches allows for inputs of varying dimension.
  const int batch_size = this->layer_param_.data_param().batch_size();
  SegmentDatum& seg_datum = *(reader_.full().peek());
  // Use data_transformer to infer the expected blob shape from anno_datum.
  vector<int> top_img_shape =
      this->data_transformer_->InferBlobShape(seg_datum.imgdata());
  this->transformed_data_.Reshape(top_img_shape);
  // Reshape batch according to the batch_size.
  top_img_shape[0] = batch_size;
  batch->data_.Reshape(top_img_shape);

  Dtype* top_img_data = batch->data_.mutable_cpu_data();
  Dtype* top_seg_data = batch->label_.mutable_cpu_data();

  for (int item_id = 0; item_id < batch_size; ++item_id) {
    timer.Start();
            SegmentDatum& seg_datum = *(reader_.full().pop("Waiting for data"));
            read_time += timer.MicroSeconds();
            timer.Start();
            SegmentDatum sampled_datum;
            sampled_datum.CopyFrom(seg_datum);

            int offset_img = batch->data_.offset(item_id);
            int offset_seg = batch->label_.offset(item_id);
            this->transformed_img_data_.set_cpu_data(top_img_data + offset_img);
            this->transformed_seg_data_.set_cpu_data(top_seg_data + offset_seg);

  LOG(INFO)<<"THE ERR always followed";
        this->data_transformer_->Transform(sampled_datum.imgdata(),   
        						&(this->transformed_img_data_), 
        						sampled_datum.maskdata(),   
        						&(this->transformed_seg_data_));
        
    trans_time += timer.MicroSeconds();

    reader_.free().push(const_cast<SegmentDatum*>(&seg_datum));
  }

  timer.Stop();
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(SegmentedDataLayer);
REGISTER_LAYER_CLASS(SegmentedData);

}  // namespace caffe
