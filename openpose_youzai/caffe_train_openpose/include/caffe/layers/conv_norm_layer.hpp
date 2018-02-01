#ifndef CAFFE_CONV_NORM_LAYER_HPP_
#define CAFFE_CONV_NORM_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/base_conv_layer.hpp"

namespace caffe {

template <typename Dtype>
class ConvolutionNormLayer : public BaseConvolutionLayer<Dtype> {
 public:

  explicit ConvolutionNormLayer(const LayerParameter& param)
      : BaseConvolutionLayer<Dtype>(param) {}

  virtual inline const char* type() const { return "ConvolutionNorm"; }
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual inline bool reverse_dimensions() { return false; }
  virtual void compute_output_shape();
private:
    Blob<Dtype> bottom_quantize_;
    Blob<Dtype> top_dequantize_;
};

}  // namespace caffe

#endif  // CAFFE_CONV_NORM_LAYER_HPP_
