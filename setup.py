# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

from distutils import sysconfig
from functools import wraps
import os
import shutil
import subprocess

from setuptools import find_packages
from setuptools import setup
from setuptools.command.install import install


def _chdir_back(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        current_dir = os.getcwd()
        result = f(*args, **kwargs)
        os.chdir(current_dir)
        return result

    return wrapped


class AdditionalDependencyInstallCommand(install):
    """Post-installation for installation mode."""

    def install_object_detection(self):
        print("clone https://github.com/tensorflow/models/ "
              "with object_detection")
        self._object_detection_clone()
        print("compile protos")
        self._object_detection_compile_protos()
        print("move out directories")
        self._object_detection_finalize()

    @_chdir_back
    def _object_detection_clone(self):
        os.chdir(self.site_packages_path)
        subprocess.check_call(
            """
            git clone --no-checkout https://github.com/tensorflow/models/
            cd models
            git config core.sparseCheckout true
            echo "research/object_detection/*"> .git/info/sparse-checkout
            git checkout 8044453
            """,
            shell=True
        )

    @_chdir_back
    def _object_detection_compile_protos(self):
        os.chdir(os.path.join(self.site_packages_path, "models"))
        subprocess.check_call(
            """
            cd research
            protoc object_detection/protos/*.proto --python_out=.
            """,
            shell=True
        )

    @_chdir_back
    def _object_detection_finalize(self):
        os.chdir(self.site_packages_path)
        shutil.move("models/research/object_detection", "object_detection")
        shutil.rmtree("models")

    @property
    def site_packages_path(self):
        return sysconfig.get_python_lib()

    def run(self):
        self.install_object_detection()
        install.run(self)


version = open("ncgenes7/VERSION", "r").read().strip()

setup(
    name='ncgenes7',
    packages=find_packages(),
    description='ncgenes7 library',
    url="https://github.com/AEV/ncgenes7",
    version=version,
    author='Oleksandr Vorobiov',
    author_email='oleksandr.vorobiov@audi.de',
    keywords=['ncgenes7', 'nucleus7', 'deep learning'],
    install_requires=[
        'nucleus7 @ git+https://github.com/AEV/nucleus7@master',
        'Cython==0.29.5',
        'opencv-python>=4.0.0.21',
        'scikit-image==0.14.2',
        'pypng>=0.0.18',
        'scikit-learn>=0.18.1',
        'Pillow>=5.2.0',
        'contextlib2>=0.5.0',
        'lxml>=4.1.1',
        'pandas>=0.24.1',
    ],
    cmdclass={
        "install_additional_deps": AdditionalDependencyInstallCommand,
    },
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MPL 2.0 License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
    data_files=[('', ['ncgenes7/VERSION'])],
)
