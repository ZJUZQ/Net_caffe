#include <algorithm>
#include <vector>
#include <float.h>
#include "caffe/layers/batch_norm_int_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void BatchNormIntLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int num = bottom[0]->shape(0);
  int spatial_dim = bottom[0]->count()/(channels_*num);

  if (bottom[0] != top[0]) {
    caffe_copy(bottom[0]->count(), bottom_data, top_data);
  }
#if 1
  caffe_gpu_scale<Dtype>(top[0]->count(), 256,
                  bottom[0]->gpu_data(), top[0]->mutable_gpu_data());

  Dtype *pTop = top[0]->mutable_cpu_data();
  for (int i=0; i<top[0]->count(); i++)
  {
      int val = int(pTop[i]);
      //val = val >0? val: 0;
      //val = val <(intV_-1)? val:(intV_-1);
      pTop[i]=val;
  }    
#else
  if (use_global_stats_) {
    // use the stored mean/variance estimates.
    const Dtype scale_factor = this->blobs_[3]->cpu_data()[0] == 0 ?
        0 : 1 / this->blobs_[3]->cpu_data()[0];
    caffe_gpu_scale(mean_.count(), scale_factor,
                    this->blobs_[0]->gpu_data(), mean_.mutable_gpu_data());
    caffe_gpu_scale(mean_.count(), scale_factor,
                    this->blobs_[1]->gpu_data(), min_.mutable_gpu_data());
    caffe_gpu_scale(mean_.count(), scale_factor,
                    this->blobs_[2]->gpu_data(), max_.mutable_gpu_data());
    // compute the range
    caffe_gpu_sub<Dtype>(channels_,
                         max_.gpu_data(),min_.gpu_data(),
                         range_.mutable_gpu_data());
  } else {
    // compute mean 
    caffe_gpu_gemv<Dtype>(CblasNoTrans, channels_ * num, spatial_dim,
        1. / (num * spatial_dim), bottom_data,
        spatial_sum_multiplier_.gpu_data(), 0.,
        num_by_chans_.mutable_gpu_data());
    caffe_gpu_gemv<Dtype>(CblasTrans, num, channels_, 1.,
        num_by_chans_.gpu_data(), batch_sum_multiplier_.gpu_data(), 0.,
        mean_.mutable_gpu_data());
                
    // compute min max on CPU , move to gpu
    const Dtype*  pBottom_CPU = bottom[0]->cpu_data();
    Dtype*  pMin = min_.mutable_cpu_data();
    Dtype*  pMax = max_.mutable_cpu_data();
    for (int c=0; c<channels_; c++)
    {
        Dtype minV=Dtype(FLT_MAX), maxV=Dtype(FLT_MIN);
        const Dtype* pTMP = pBottom_CPU + spatial_dim;
        for (int n=0; n<num; n++)
        {       
            std::vector<Dtype> vec(pTMP, pTMP+spatial_dim);
            std::sort(vec.begin(), vec.begin()+spatial_dim);
            if (vec[0]<minV) minV = vec[0];
            if (vec[spatial_dim-1]>maxV) maxV = vec[spatial_dim-1];
            pTMP += channels_*spatial_dim;
        }
        pMin[c] = minV;
        pMax[c] = maxV;
    }                
    // compute and save moving average
    this->blobs_[3]->mutable_cpu_data()[0] *= moving_average_fraction_;
    this->blobs_[3]->mutable_cpu_data()[0] += 1;
    caffe_gpu_axpby(mean_.count(), Dtype(1), mean_.gpu_data(),
        moving_average_fraction_, this->blobs_[0]->mutable_gpu_data());
    caffe_gpu_axpby(min_.count(), Dtype(1), min_.gpu_data(),
        moving_average_fraction_, this->blobs_[1]->mutable_gpu_data());
    caffe_gpu_axpby(max_.count(), Dtype(1), max_.gpu_data(),
        moving_average_fraction_, this->blobs_[2]->mutable_gpu_data());

    //compute range
    caffe_gpu_sub<Dtype>(channels_,
                         max_.gpu_data(),min_.gpu_data(),
                         range_.mutable_gpu_data());
  }

  // subtract min
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
                        batch_sum_multiplier_.gpu_data(), min_.gpu_data(), 0.,
                        num_by_chans_.mutable_gpu_data());
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
                        spatial_dim, 1, -1, num_by_chans_.gpu_data(),
                        spatial_sum_multiplier_.gpu_data(), 1., top_data);

  // normalize range
  caffe_gpu_add_scalar(range_.count(), eps_, range_.mutable_gpu_data());
  caffe_gpu_scale<Dtype>(range_.count(), 1.f/(intV_-1), range_.gpu_data(), 
                         range_.mutable_gpu_data());
                         
  // replicate range to input size
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
      batch_sum_multiplier_.gpu_data(), range_.gpu_data(), 0.,
      num_by_chans_.mutable_gpu_data());
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
      spatial_dim, 1, 1., num_by_chans_.gpu_data(),
      spatial_sum_multiplier_.gpu_data(), 0., temp_.mutable_gpu_data());
  caffe_gpu_div(temp_.count(), top_data, temp_.gpu_data(), top_data);
  // TODO(cdoersch): The caching is only needed because later in-place layers
  //                 might clobber the data.  Can we skip this if they won't?
  Dtype *pTop = top[0]->mutable_cpu_data();
  for (int i=0; i<top[0]->count(); i++)
  {
      int val = int(pTop[i]);
      val = val >0? val: 0;
      val = val <(intV_-1)? val:(intV_-1);
      pTop[i] = val;
  }    
#endif
  caffe_copy(x_norm_.count(), top_data,
      x_norm_.mutable_gpu_data());

}

template <typename Dtype>
void BatchNormIntLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                            const vector<bool>& propagate_down,
                                            const vector<Blob<Dtype>*>& bottom) {
#if 1
  const Dtype* top_diff;
  if (bottom[0] != top[0]) {
    top_diff = top[0]->gpu_diff();
  } else {
    caffe_copy(x_norm_.count(), top[0]->gpu_diff(), x_norm_.mutable_gpu_diff());
    top_diff = x_norm_.gpu_diff();
  }
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
//  if (use_global_stats_) {
  if (1)
  {
    caffe_gpu_scale<Dtype>(top[0]->count(), 256.0f,
                    top_diff, bottom_diff);
    return;
  }
#else
  const Dtype* top_diff;
  if (bottom[0] != top[0]) {
    top_diff = top[0]->gpu_diff();
  } else {
    caffe_copy(x_norm_.count(), top[0]->gpu_diff(), x_norm_.mutable_gpu_diff());
    top_diff = x_norm_.gpu_diff();
  }
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
//  if (use_global_stats_) {
  if (1)
  {
    caffe_gpu_div(temp_.count(), top_diff, temp_.gpu_data(), bottom_diff);
    return;
  }
#endif
}

INSTANTIATE_LAYER_GPU_FUNCS(BatchNormIntLayer);


}  // namespace caffe
