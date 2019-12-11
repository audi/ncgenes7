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

echo Add requirements from ${DEPS_DIR} to PYTHONPATH

OBJECT_DETECTION_DIR=${DEPS_DIR}/models/research
echo 1. Add object_detection ${OBJECT_DETECTION_DIR}
export PYTHONPATH=$OBJECT_DETECTION_DIR:${PYTHONPATH}

echo Done
