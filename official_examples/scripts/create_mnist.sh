#!/usr/bin/env sh
# This script converts the mnist data into lmdb/leveldb format,
# depending on the value assigned to $BACKEND.

set -e # Every script you write should include set -e at the top. This tells bash that it should exit the script if any statement returns a non-true return value. 

DATA=../data/mnist
BUILD=$1/build/examples/mnist

BACKEND="lmdb"

echo "Creating ${BACKEND}..."

rm -rf mnist_train_${BACKEND}
rm -rf mnist_test_${BACKEND}

$BUILD/convert_mnist_data.bin $DATA/train-images-idx3-ubyte \
  $DATA/train-labels-idx1-ubyte mnist_train_${BACKEND} --backend=${BACKEND}
$BUILD/convert_mnist_data.bin $DATA/t10k-images-idx3-ubyte \
  $DATA/t10k-labels-idx1-ubyte mnist_test_${BACKEND} --backend=${BACKEND}

echo "Done."
