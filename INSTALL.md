Installation
============

- [Dependencies download](#dependencies-download)
- [Package installation](#package-installation)
- [Local installation](#local-installation)
- [Build documentation](#api-documentation)

[requirements]: ./requirements.txt
[requirements_doc]: ./requirements_docs.txt
[sphinx-install]: https://www.sphinx-doc.org/en/master/usage/installation.html
[object_detection]: https://github.com/tensorflow/models/tree/master/research/object_detection

## Dependencies download <a name="dependencies-download"></a>

For some modules to work, you need to download some packages from github,
since they cannot be installed using some package manager.
Supported versions are controlled over commit hashes, but you
can try the most recent ones if you want so. Following table shows what and
when you need: 

| nucleotides | required dependency github link | latest supported commit|  
| :--------: | :---------: | :---------: |
| object detection related nucleotides, like plugins, losses etc.  | [github][object_detection] | 8044453 |

Default download and activation, e.g. adding the modules to PYTHONPATH can be
done using following scripts:

```bash
chmod 750 download_dependencies.sh
./download_dependencies.sh
source activate_dependencies.sh
```

This will download dependencies in the folder `../ncgenes7_dependencies/`
and add this folder to PYTHONPATH. You need to have the packages inside
of PYTHONPATH for both types of installation / activation.

Also you can install the dependencies directly to you site-packages using:

```bash
python3 setup.py install_additional_deps
```

This is best to use with virtualenv, like shown below.

**Important**: object_detection has its own dependencies, which will not be
installed by this installation. So refer to [github][object_detection]
installation guide. Also you need to have the protobuf installed prior the
`./download_dependencies.sh` or `python3 setup.py install_additional_deps`.

## Package installation <a name="package-installation"></a>

Make sure, that nucleus7 already installed with appropriate version!

To install it, just type (remove virtual environment commands if you want to
install in the default environment):

```bash
virtualenv --python=python3.6 ~/.env-ncgenes7
source ~/.env-ncgenes7/bin/activate
python3 setup.py install
```

You are free to select the tensorflow version, but it was tested with
tensorflow(-gpu) >=1.11, <2.0

## Local installation <a name="local-installation"></a>

First you need to install the [requirements][requirements]:

```bash
pip3 install -r requirements.txt
```

or [documentation requirements][requirements_doc] (includes requirements for docs build):

```bash
pip3 install -r requirements_docs.txt
```

To add ncgenes7 to `PYTHONPATH` just source the `activate.sh` which will
setup everything inside the current shell session:

```bash
source activate.sh
```

## Build documentation <a name="api-documentation"></a>

Most important methods and classes are documented inside of its docstring using
[NumpyDoc][numpydoc] format. So we can build
the documentation using sphinx.
So first [install sphinx][sphinx-install] if you don't have it,
and the build the docs:

```bash
cd docs
make html
```
