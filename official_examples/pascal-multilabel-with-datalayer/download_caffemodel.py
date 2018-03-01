#!/usr/bin/env python
import os
import sys
import time
import yaml
import urllib
import hashlib
import argparse

def reporthook(count, block_size, total_size):
    """
    From http://blog.moleculea.com/2012/10/04/urlretrieve-progres-indicator/
    """
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = (time.time() - start_time) or 0.01
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                    (percent, progress_size / (1024 * 1024), speed, duration))
    sys.stdout.flush()


def download_caffemodel(dirname):
    frontmatter = {'name': 'BVLC CaffeNet Model',
                   'caffemodel': 'bvlc_reference_caffenet.caffemodel',
                   'caffemodel_url': 'http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel',
                   'sha1': '4c8d77deb20ea792f84eb5e6d0a11ca0a8660a46',
                  }
    model_filename = os.path.join(dirname, frontmatter['caffemodel'])

    # Closure-d function for checking SHA1.
    def model_checks_out(filename=model_filename, sha1=frontmatter['sha1']):
        with open(filename, 'rb') as f:
            return hashlib.sha1(f.read()).hexdigest() == sha1

    # Check if model exists.
    if os.path.exists(model_filename) and model_checks_out():
        print("Model already exists.")
        sys.exit(0)

    # Download and verify model.
    urllib.urlretrieve(frontmatter['caffemodel_url'], model_filename, reporthook)
    if not model_checks_out():
        print('ERROR: model did not download correctly! Run this again.')
        sys.exit(1)