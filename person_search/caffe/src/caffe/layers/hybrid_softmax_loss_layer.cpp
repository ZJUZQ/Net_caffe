#include <algorithm>
#include <functional>
#include <utility>
#include <cfloat>
#include <vector>
#include <numeric>

#include "caffe/filler.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"


namespace caffe {

template <typename Dtype>
void HybridSoftmaxLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);

  num_classes_ = this->layer_param_.hybrid_softmax_loss_param().num_classes();
  ul_pool_size_ = this->layer_param_.hybrid_softmax_loss_param().unlabeled_pool_size();
  ul_score_weight_ = this->layer_param_.hybrid_softmax_loss_param().unlabeled_score_weight();
  ul_pool_tail_ = 0;
  ul_pool_full_ = false;
  num_sampling_ = this->layer_param_.hybrid_softmax_loss_param().random_sampling_num();
  sampling_policy_ = this->layer_param_.hybrid_softmax_loss_param().random_sampling_policy();
  if (sampling_policy_ != "random") {
    LOG(FATAL) << "Cannot recognize sampling policy " << sampling_policy_;
  }
  CHECK_LE(num_sampling_, num_classes_)
      << "Number of classes to be sampled should be small than "
      << "total number of classes " << num_classes_;

  num_ = bottom[0]->shape(0);
  dim_ = bottom[0]->count(1);

  // prefill the index for shuffling
  index_vec_.resize(num_classes_);
  for (int i = 0; i < num_classes_; ++i)
    index_vec_[i] = i;

  if (this->blobs_.size() > 0) {
    CHECK_EQ(this->blobs_[0]->shape(0), num_classes_ + ul_pool_size_);
    CHECK_EQ(this->blobs_[0]->count(1), dim_);
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(1);
    // weight for labeled data
    vector<int> weight_shape(2);
    weight_shape[0] = num_classes_ + ul_pool_size_;
    weight_shape[1] = dim_;
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.hybrid_softmax_loss_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
  }

  LayerParameter softmax_param(this->layer_param_);
  softmax_param.set_type("Softmax");
  softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(&softmax_bottom_);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&softmax_top_);
  vector<int> softmax_bottom_shape(2);
  softmax_bottom_shape[0] = num_;
  softmax_bottom_shape[1] = num_sampling_ + ul_pool_size_;
  softmax_bottom_.Reshape(softmax_bottom_shape);
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);

  if (!this->layer_param_.loss_param().has_normalization() &&
      this->layer_param_.loss_param().has_normalize()) {
    normalization_ = this->layer_param_.loss_param().normalize() ?
                     LossParameter_NormalizationMode_VALID :
                     LossParameter_NormalizationMode_BATCH_SIZE;
  } else {
    normalization_ = this->layer_param_.loss_param().normalization();
  }

  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void HybridSoftmaxLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  num_ = bottom[0]->shape(0);
  CHECK_EQ(dim_, bottom[0]->count(1))
    << "Input feature dimension cannot be changed.";
  CHECK_EQ(num_, bottom[1]->count())
    << "Number of input labels must match input features.";

  fc_bottom_.ReshapeLike(*(bottom[0]));
  vector<int> shape(2);
  shape[0] = num_;
  shape[1] = num_classes_ + ul_pool_size_;
  fc_top_.Reshape(shape);
  // We do not reshape the softmax layer here but will do it in the forward
}

template <typename Dtype>
void HybridSoftmaxLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  // copy labeled data to fc_bottom
  Dtype* fc_bottom_data = fc_bottom_.mutable_cpu_data();
  lb_indices_.clear();
  for (int i = 0; i < num_; ++i) {
    const int label_value = static_cast<int>(label[i]);
    if (0 <= label_value && label_value < num_classes_) {
      caffe_copy(dim_, bottom_data + i * dim_,
                 fc_bottom_data + lb_indices_.size() * dim_);
      lb_indices_.push_back(i);
    }
  }
  // update unlabeled pool
  if (ul_pool_size_ > 0) {
    Dtype* ul_pool_data = this->blobs_[0]->mutable_cpu_data() +
                          num_classes_ * dim_;
    ul_indices_.clear();
    for (int i = 0; i < num_; ++i) {
      const int label_value = static_cast<int>(label[i]);
      if (label_value != -1) continue;
      caffe_cpu_scale(dim_, ul_score_weight_,
                      bottom_data + i * dim_,
                      ul_pool_data + ul_pool_tail_ * dim_);
      ul_indices_.push_back(make_pair(i, ul_pool_tail_));
      if (++ul_pool_tail_ >= ul_pool_size_) {
        ul_pool_full_ = true;
        ul_pool_tail_ = 0;
      }
    }
  }
  // compute fc output scores
  const int M = static_cast<int>(lb_indices_.size());
  const int N1 = num_classes_ + ul_pool_size_;
  const int K = dim_;
  if (M == 0) {
    top[0]->mutable_cpu_data()[0] = 0;
    return;
  }
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M, N1, K,
      (Dtype)1., fc_bottom_.cpu_data(), this->blobs_[0]->cpu_data(),
      (Dtype)0., fc_top_.mutable_cpu_data());
  // random sample a subset of labeled classes for each data sample
  RandomSample(bottom);
  // compute softmax probability
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  /*
  // for tuning the unlabeled score weight
  for (int i = 0; i < M; ++i) {
    const Dtype* score = softmax_bottom_.cpu_data() + i * N;
    Dtype max_score_1 = (Dtype)(-FLT_MAX);
    for (int j = 0; j < num_classes_; ++j)
      max_score_1 = std::max(max_score_1, score[j]);
    Dtype max_score_2 = (Dtype)(-FLT_MAX);
    for (int j = num_classes_; j < N; ++j)
      max_score_2 = std::max(max_score_2, score[j]);
    printf("%.6lf   %.6lf\n", max_score_1, max_score_2);
  }
  */
  // compute nll loss
  const Dtype* prob = softmax_top_.cpu_data();
  const int N2 = softmax_top_.shape(1);
  Dtype loss = 0;
  for (int i = 0; i < M; ++i) {
    // after random sampling, the target label is 0 for all the data samples
    loss -= log(std::max(prob[i * N2], (Dtype)FLT_MIN));
  }
  top[0]->mutable_cpu_data()[0] = loss / std::max(1, M);
}

template <typename Dtype>
void HybridSoftmaxLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* prob = softmax_top_.cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  caffe_set(bottom[0]->count(), (Dtype)0., bottom_diff);
  // propagate through softmax layer
  const int M = static_cast<int>(lb_indices_.size());
  const int K = num_sampling_;
  if (M == 0) { return; }
  Dtype* softmax_bottom_diff = softmax_bottom_.mutable_cpu_diff();
  caffe_copy(softmax_top_.count(), prob, softmax_bottom_diff);
  const int N2 = softmax_bottom_.shape(1);
  for (int i = 0; i < M; ++i)
    softmax_bottom_diff[i * N2] -= 1;
  const Dtype loss_weight = top[0]->cpu_diff()[0];
  caffe_scal(softmax_bottom_.count(), loss_weight / M, softmax_bottom_diff);
  // copy the diff from softmax_bottom to corresponding fc_top
  const int N1 = fc_top_.shape(1);
  Dtype* fc_top_diff = fc_top_.mutable_cpu_diff();
  caffe_set(fc_top_.count(), (Dtype)0, fc_top_diff);
  for (int i = 0; i < M; ++i) {
    const vector<int>& s = sampled_index_[i];
    for (int j = 0; j < s.size(); ++j)
      fc_top_diff[i*N1 + s[j]] = softmax_bottom_diff[i*N2 + j];
    caffe_copy(N2 - K, softmax_bottom_diff + i*N2 + K,
               fc_top_diff + i*N1 + num_classes_);
  }

  if (this->param_propagate_down_[0] ||
        (propagate_down[0] && !ul_indices_.empty())) {
    // propagate to weight for labeled classes
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N1, dim_, M,
        (Dtype)1., fc_top_.cpu_diff(), fc_bottom_.cpu_data(),
        (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
    // copy corresponding diff to the unlabeled samples in this minibatch
    const Dtype* w_diff = this->blobs_[0]->cpu_diff();
    for (int i = 0; i < ul_indices_.size(); ++i) {
      const int minibatch_index = ul_indices_[i].first;
      const int w_index = num_classes_ + ul_indices_[i].second;
      caffe_cpu_scale(dim_, ul_score_weight_,
                      w_diff + w_index * dim_,
                      bottom_diff + minibatch_index * dim_);
    }
    if (!this->param_propagate_down_[0]) {
      caffe_set(this->blobs_[0]->count(), (Dtype)0.,
                this->blobs_[0]->mutable_cpu_diff());
    } else if (ul_pool_size_ > 0) {
      // set diff of weight for unlabeled classes to zero
      caffe_set(ul_pool_size_ * dim_, (Dtype)0.,
                this->blobs_[0]->mutable_cpu_diff() + num_classes_ * dim_);
    }
  }
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    // propagate to labeled samples
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M, dim_, N1,
        (Dtype)1., fc_top_.cpu_diff(), this->blobs_[0]->cpu_data(),
        (Dtype)0., fc_bottom_.mutable_cpu_diff());
    // copy back to minibatch
    const Dtype* fc_bottom_diff = fc_bottom_.cpu_diff();
    for (int i = 0; i < M; ++i) {
      const int minibatch_index = lb_indices_[i];
      caffe_copy(dim_, fc_bottom_diff + i * dim_,
                 bottom_diff + minibatch_index * dim_);
    }
  }
}

template <typename Dtype>
void HybridSoftmaxLossLayer<Dtype>::RandomSample(
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* label = bottom[1]->cpu_data();
  const int M = lb_indices_.size();
  const int K = num_sampling_;
  // generate the sampled index
  sampled_index_.resize(M, vector<int>(K, 0));
  for (int i = 0; i < M; ++i) {
    const int minibatch_index = lb_indices_[i];
    const int label_value = static_cast<int>(label[minibatch_index]);
    CHECK_GE(label_value, 0);
    CHECK_LT(label_value, num_classes_);
    // put the ground truth class to the first
    sampled_index_[i][0] = label_value;
    // randomly choose K - 1 other classes
    shuffle(index_vec_.begin(), index_vec_.end());
    for (int j = 0, k = 1; j < index_vec_.size() && k < K; ++j) {
      if (index_vec_[j] == label_value) continue;
      sampled_index_[i][k++] = index_vec_[j];
    }
  }
  // copy scores from fc_top to softmax_bottom
  const int N1 = fc_top_.shape(1);
  const int N2 = K + (ul_pool_full_ ? ul_pool_size_ : ul_pool_tail_);
  vector<int> shape(2);
  shape[0] = M;
  shape[1] = N2;
  softmax_bottom_.Reshape(shape);
  const Dtype* fc_top = fc_top_.cpu_data();
  Dtype* sm_bottom = softmax_bottom_.mutable_cpu_data();
  for (int i = 0; i < M; ++i) {
    const vector<int>& s = sampled_index_[i];
    // labeled class scores
    for (int j = 0; j < s.size(); ++j)
      sm_bottom[i * N2 + j] = fc_top[i * N1 + s[j]];
    // unlabeled class scores
    caffe_copy(N2 - K, fc_top + i*N1 + num_classes_,
               sm_bottom + i*N2 + K);
  }
}

INSTANTIATE_CLASS(HybridSoftmaxLossLayer);
REGISTER_LAYER_CLASS(HybridSoftmaxLoss);

}  // namespace caffe
