#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/inner_product_norm_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void scale8_fc(const int nthreads,
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
__global__ void scale8_fc_back(const int nthreads,
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
__global__ void descale8_fc(const int nthreads,
                            const Dtype* const in,
                            Dtype* out,
                            float int_scale) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    out[index] = in[index]/int_scale;
  }
}

template <typename Dtype>
void InnerProductNormLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype* weight = this->blobs_[0]->gpu_data();

  Dtype* bot_quan_data = bottom_quantize_.mutable_gpu_data();
  Dtype* top_dequan_data = top_dequantize_.mutable_gpu_data();

  if (M_ == 1) {
    //quantize to uint8
    scale8_fc<Dtype><<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(bottom[0]->count(),
                         bottom_data, bot_quan_data, this->int_scale_);

    caffe_gpu_gemv<Dtype>(CblasNoTrans, N_, K_, (Dtype)1.,
                         weight, bot_quan_data, (Dtype)0., top_dequan_data);
                               // dequantize to float
    descale8_fc<Dtype><<<CAFFE_GET_BLOCKS(top[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(top[0]->count(),
                      top_dequan_data, top_data, this->int_scale_);

    if (bias_term_)
      caffe_gpu_axpy<Dtype>(N_, bias_multiplier_.cpu_data()[0],
                            this->blobs_[1]->gpu_data(), top_data);
  } else {
    scale8_fc<Dtype><<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(bottom[0]->count(),
                         bottom_data, bot_quan_data, this->int_scale_);
    caffe_gpu_gemm<Dtype>(CblasNoTrans,
                          transpose_ ? CblasNoTrans : CblasTrans,
                          M_, N_, K_, (Dtype)1.,
                          bot_quan_data, weight, (Dtype)0., top_dequan_data);
    descale8_fc<Dtype><<<CAFFE_GET_BLOCKS(top[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(top[0]->count(),
                      top_dequan_data, top_data, this->int_scale_);                          
    if (bias_term_)
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
                            bias_multiplier_.gpu_data(),
                            this->blobs_[1]->gpu_data(), (Dtype)1., top_data);
  }
}

template <typename Dtype>
void InnerProductNormLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* bot_quan_data = bottom_quantize_.mutable_gpu_data();
    // Gradient with respect to weight

    scale8_fc_back<Dtype><<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(bottom[0]->count(),
                         bottom_data, bot_quan_data, this->int_scale_);
    if (transpose_) {
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          K_, N_, M_,
          (Dtype)1., bot_quan_data, top_diff,
          (Dtype)1., this->blobs_[0]->mutable_gpu_diff());
    } else {
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          N_, K_, M_,
          (Dtype)1., top_diff, bot_quan_data,
          (Dtype)1., this->blobs_[0]->mutable_gpu_diff());
    }
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    // Gradient with respect to bias
    caffe_gpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
        bias_multiplier_.gpu_data(), (Dtype)1.,
        this->blobs_[1]->mutable_gpu_diff());
  }
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    // Gradient with respect to bottom data
    if (transpose_) {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
          M_, K_, N_,
          (Dtype)1., top_diff, this->blobs_[0]->gpu_data(),
          (Dtype)0., bottom[0]->mutable_gpu_diff());
    } else {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
          M_, K_, N_,
         (Dtype)1., top_diff, this->blobs_[0]->gpu_data(),
         (Dtype)0., bottom[0]->mutable_gpu_diff());
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(InnerProductNormLayer);

}  // namespace caffe
