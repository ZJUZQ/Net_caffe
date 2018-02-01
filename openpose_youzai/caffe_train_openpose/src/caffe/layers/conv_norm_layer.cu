#include <vector>

#include "caffe/layers/conv_norm_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void scale8(const int nthreads,
                      const Dtype* const in,
                      Dtype* out,
                      float int_scale) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    Dtype tmp = in[index];                          
    tmp = int(tmp*int_scale+0.5f);
    tmp = tmp<0? 0:tmp;
    tmp = tmp>255?255:tmp;
    out[index] = tmp;
  }
}

template <typename Dtype>
__global__ void scale8_back(const int nthreads,
                            const Dtype* const in,
                            Dtype* out,
                            float int_scale) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    Dtype tmp = in[index];                          
    tmp = int(tmp*int_scale+0.5f);
    tmp = tmp<0? 0:tmp;
    tmp = tmp>255?255:tmp;
    out[index] = tmp/int_scale;
  }
}

template <typename Dtype>
__global__ void descale8(const int nthreads,
                        const Dtype* const in,
                        Dtype* out,
                        float int_scale) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    out[index] = in[index]/int_scale;
  }
}

template <typename Dtype>
void ConvolutionNormLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();
    Dtype* bot_quan_data = bottom_quantize_.mutable_gpu_data();
    Dtype* top_dequan_data = top_dequantize_.mutable_gpu_data();
    for (int n = 0; n < this->num_; ++n) {
      //quantize to uint8
      scale8<Dtype><<<CAFFE_GET_BLOCKS(this->bottom_dim_), CAFFE_CUDA_NUM_THREADS>>>(this->bottom_dim_,
                         bottom_data + n * this->bottom_dim_,
                         bot_quan_data, this->int_scale_);
      this->forward_gpu_gemm(bot_quan_data, weight,
                             top_dequan_data);
      // dequantize to float
      descale8<Dtype><<<CAFFE_GET_BLOCKS(this->top_dim_), CAFFE_CUDA_NUM_THREADS>>>(this->top_dim_,
                      top_dequan_data,
                      top_data + n * this->top_dim_,
                      this->int_scale_);
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->gpu_data();
        this->forward_gpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
}

template <typename Dtype>
void ConvolutionNormLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_gpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      const Dtype* bottom_data = bottom[i]->gpu_data();
      Dtype* bot_quan_data = bottom_quantize_.mutable_gpu_data();
      Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
         scale8_back<Dtype><<<CAFFE_GET_BLOCKS(this->bottom_dim_), CAFFE_CUDA_NUM_THREADS>>>(this->bottom_dim_,
                         bottom_data + n * this->bottom_dim_,
                         bot_quan_data, this->int_scale_);
          this->weight_gpu_gemm(bot_quan_data,
              top_diff + n * this->top_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_gpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ConvolutionNormLayer);

}  // namespace caffe
