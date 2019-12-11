#!/usr/bin/env bash
# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

WD="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
DEPS_DIR=$(dirname ${WD})/ncgenes7_dependencies

echo Download the requirements to ${DEPS_DIR}

echo 1. Create folder
mkdir -p ${DEPS_DIR}

cd ${DEPS_DIR}
echo `pwd`

echo 2. object_detection API for tensorflow from google and slim
echo from https://github.com/tensorflow/models/
echo 2.1 Sparse checkout

git clone --no-checkout https://github.com/tensorflow/models/
cd models
git config core.sparseCheckout true
echo "research/object_detection/*"> .git/info/sparse-checkout
git checkout 8044453

echo 2.2 Compile protobufs
cd research
protoc object_detection/protos/*.proto --python_out=.
cd ${DEPS_DIR}

echo Done
