#include <algorithm>
#include <functional>
#include <utility>
#include <cfloat>
#include <vector>

#include "caffe/common_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
void TargetSamplingLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  K_ = this->layer_param_.target_sampling_param().num_sampling();
  M_ = bottom[0]->shape(0);
  D_ = bottom[1]->count(1);

  indices_.resize(D_);
  for (int i = 0; i < D_; ++i) indices_[i] = i;
  sampled_.resize(M_, vector<int>(K_, 0));
}

template <typename Dtype>
void TargetSamplingLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int new_M = bottom[0]->shape(0);
  const int new_D = bottom[0]->count(1);
  if (M_ != new_M) {
    M_ = new_M;
    sampled_.resize(M_, vector<int>(K_, 0));
  }
  if (D_ != new_D) {
    D_ = new_D;
    indices_.resize(D_);
    for (int i = 0; i < D_; ++i) indices_[i] = i;
  }
  CHECK_GE(D_, K_)
      << "Input feature dim must be greater or equal to "
      << "the dimension to be sampled";
  vector<int> shape(2);
  shape[0] = M_;
  shape[1] = K_;
  top[0]->Reshape(shape);
  top[1]->ReshapeLike(*(bottom[1]));
}

template <typename Dtype>
void TargetSamplingLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype* top_label = top[1]->mutable_cpu_data();
  caffe_set(top[0]->count(), (Dtype)0., top_data);
  for (int i = 0; i < M_; ++i) {
    const int label = static_cast<int>(bottom_label[i]);
    vector<int>& s = sampled_[i];
    if (0 <= label && label < D_) {
      // put the ground truth class to the first
      s[0] = label;
      // randomly choose K_ - 1 other classes
      shuffle(indices_.begin(), indices_.end());
      for (int j = 0, k = 1; j < indices_.size() && k < K_; ++j) {
        if (indices_[j] == label) continue;
        s[k++] = indices_[j];
      }
      // copy data
      for (int k = 0; k < K_; ++k)
        top_data[i * K_ + k] = bottom_data[i * D_ + s[k]];
      // assign label
      top_label[i] = (Dtype)0.;
    } else {
      top_label[i] = (Dtype)(-1.);
    }
  }
}

template <typename Dtype>
void TargetSamplingLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* top_label = top[1]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_set(bottom[0]->count(), (Dtype)0., bottom_diff);
    for (int i = 0; i < M_; ++i) {
      const int label = static_cast<int>(top_label[i]);
      if (label == -1) continue;
      const vector<int>& s = sampled_[i];
      for (int k = 0; k < K_; ++k)
        bottom_diff[i * D_ + s[k]] = top_diff[i * K_ + k];
    }
  }
}

INSTANTIATE_CLASS(TargetSamplingLayer);
REGISTER_LAYER_CLASS(TargetSampling);

}  // namespace caffe
