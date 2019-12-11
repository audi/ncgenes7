# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

.PHONY: test, test-local, test-local-fast
NCGENES7_DIR=`pwd`

export PYTHONPATH := ${NCGENES7_DIR}:${PYTHONPATH}

test:
	tox

test-local:
	pytest --log-level ERROR --disable-pytest-warnings tests

test-local-fast:
	python3 -m pytest -m "not slow" --log-level ERROR --disable-pytest-warnings tests
