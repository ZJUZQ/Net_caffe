#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

using boost::scoped_ptr;

namespace caffe {

template <typename Dtype>
class HybridSoftmaxLossLayerTest : public CPUDeviceTest<Dtype> {
 protected:
  HybridSoftmaxLossLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(16, 2, 1, 1)),
        blob_bottom_label_(new Blob<Dtype>(16, 1, 1, 1)),
        blob_top_loss_(new Blob<Dtype>()) {
    const Dtype dx[4] = {1, -1, -1, 1};
    const Dtype dy[4] = {1, 1, -1, -1};
    Dtype* data = blob_bottom_data_->mutable_cpu_data();
    Dtype* label = blob_bottom_label_->mutable_cpu_data();
    for (int i = 0; i < 16; ++i) {
      Dtype x = dx[i / 4] + 0.1 * dx[i % 4];
      Dtype y = dy[i / 4] + 0.1 * dy[i % 4];
      data[i * 2] = x;
      data[i * 2 + 1] = y;
      switch (i / 4) {
        case 0:
          label[i] = static_cast<Dtype>(0); break;
        case 1:
          label[i] = static_cast<Dtype>(1); break;
        case 2:
          label[i] = static_cast<Dtype>(-1); break;
        case 3:
          label[i] = static_cast<Dtype>(2); break;
      }
    }
    blob_bottom_vec_.push_back(blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_label_);
    blob_top_vec_.push_back(blob_top_loss_);
  }
  virtual ~HybridSoftmaxLossLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_label_;
    delete blob_top_loss_;
  }
  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_label_;
  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(HybridSoftmaxLossLayerTest, TestDtypes);

TYPED_TEST(HybridSoftmaxLossLayerTest, TestGradient) {
  LayerParameter layer_param;
  layer_param.add_loss_weight(1);
  layer_param.add_propagate_down(1);
  layer_param.add_propagate_down(0);
  HybridSoftmaxLossParameter* loss_param =
      layer_param.mutable_hybrid_softmax_loss_param();
  loss_param->set_num_classes(2);
  loss_param->set_unlabeled_pool_size(4);
  loss_param->mutable_weight_filler()->set_type("gaussian");
  loss_param->mutable_weight_filler()->set_std(0.001);
  loss_param->set_random_sampling_num(2);
  loss_param->set_random_sampling_policy("random");

  HybridSoftmaxLossLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

TYPED_TEST(HybridSoftmaxLossLayerTest, TestUnlabeledPool) {
  LayerParameter layer_param;
  layer_param.add_loss_weight(1);
  layer_param.add_propagate_down(1);
  layer_param.add_propagate_down(0);
  HybridSoftmaxLossParameter* loss_param =
      layer_param.mutable_hybrid_softmax_loss_param();
  loss_param->set_num_classes(2);
  loss_param->set_unlabeled_pool_size(5);
  loss_param->mutable_weight_filler()->set_type("gaussian");
  loss_param->mutable_weight_filler()->set_std(0.001);
  loss_param->set_random_sampling_num(2);
  loss_param->set_random_sampling_policy("random");

  scoped_ptr<HybridSoftmaxLossLayer<TypeParam> > layer(
      new HybridSoftmaxLossLayer<TypeParam>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(7, layer->blobs()[0]->shape(0));
  EXPECT_EQ(2, layer->blobs()[0]->shape(1));

  const TypeParam* ul_pool_data = layer->blobs()[0]->cpu_data() + 2 * 2;
  const TypeParam* bottom_data = this->blob_bottom_data_->cpu_data();
  // first forward, minibatch index [8, 9, 10, 11, -] in the pool
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const TypeParam loss_1 = this->blob_top_loss_->cpu_data()[0];
  for (int i = 0; i < 4; ++i) {
    const int minibatch_index = i + 8;
    EXPECT_EQ(bottom_data[minibatch_index * 2], ul_pool_data[i*2]);
    EXPECT_EQ(bottom_data[minibatch_index * 2 + 1], ul_pool_data[i*2+1]);
  }
  // second forward, minibatch index [9, 10, 11, 11, 8] in the pool
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const TypeParam loss_2 = this->blob_top_loss_->cpu_data()[0];
  const int minibatch_indices[5] = {9, 10, 11, 11, 8};
  for (int i = 0; i < 5; ++i) {
    const int minibatch_index = minibatch_indices[i];
    EXPECT_EQ(bottom_data[minibatch_index * 2], ul_pool_data[i*2]);
    EXPECT_EQ(bottom_data[minibatch_index * 2 + 1], ul_pool_data[i*2+1]);
  }
  EXPECT_LT(loss_1, loss_2);
}

}  // namespace caffe