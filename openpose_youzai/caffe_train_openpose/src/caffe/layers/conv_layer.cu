#include <vector>
#include <numeric>
#include "caffe/layers/conv_layer.hpp"

namespace caffe {

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();

  #if 0 //PRINT_MIN_MAX
  // sort debug
  int total_num = this->blobs_[0]->count();
  const Dtype* weights_cpu = this->blobs_[0]->cpu_data();
  Dtype* sort_data = 0;
  sort_data = new Dtype[total_num];
  memcpy(sort_data, weights_cpu, sizeof(Dtype)*total_num);
  std::vector<Dtype> myvector(sort_data, sort_data+total_num);
  std::sort (myvector.begin(), myvector.begin()+total_num);
  Dtype average = std::accumulate(myvector.begin(), myvector.end(), 0.0)/total_num; 
  std::cout<<"conv min:"<<myvector[0]<<",max_99:"<<myvector[int(total_num*0.99-1)]
            <<",max:"<<myvector[total_num-1]<<",mean:"<<average<<std::endl;
  
  delete []sort_data;
  #endif

#ifdef PRINT_MIN_MAX
    // sort debug
    const Dtype* bottom_data_cpu = bottom[i]->cpu_data();
    int total_num = this->num_*this->bottom_dim_;
    Dtype* sort_data = 0;
    sort_data = new Dtype[total_num];
    memcpy(sort_data, bottom_data_cpu, sizeof(Dtype)*total_num);
    std::vector<Dtype> myvector(sort_data, sort_data+total_num);
    std::sort (myvector.begin(), myvector.begin()+total_num);
    std::cout<<"min:"<<myvector[0]<<",max_99:"<<myvector[int(total_num*0.99-1)]<<",max:"<<myvector[total_num-1]<<std::endl;
    delete[] sort_data;
#endif

    Dtype* top_data = top[i]->mutable_gpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_, weight,
          top_data + n * this->top_dim_);
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->gpu_data();
        this->forward_gpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
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
      Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_gpu_gemm(bottom_data + n * this->bottom_dim_,
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

INSTANTIATE_LAYER_GPU_FUNCS(ConvolutionLayer);

}  // namespace caffe
