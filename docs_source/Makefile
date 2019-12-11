# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = python -msphinx
SPHINXPROJ    = ncgenes7
SOURCEDIR     = source
BUILDDIR      = build
DOCSDIR       = ../docs

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@if [ "$@" = "html" ]; then\
        # need to replace links to md files to generated html files \
        echo "Replace wrong links";\
        sed -i "s/\"[A-Za-z1-7_/.]*INSTALL.md\"/\"Installation.html\"/g" build/html/*.html ; \
        sed -i "s/\"[A-Za-z1-7_/.]*CONTRIBUTING.md\"/\"Contribution.html\"/g" build/html/*.html ; \
        \
        sed -i "s/\"[A-Za-z1-7_/.]*coco\/README.md\"/\"ProjectSamples.html#data-extraction-coco-dataset\"/g" build/html/*.html ; \
        sed -i "s/\"[A-Za-z1-7_/.]*semantic_segmentation\/README.md\"/\"ProjectSamples.html#semantic-segmentation\"/g" build/html/*.html ; \
        sed -i "s/\"[A-Za-z1-7_/.]*object_detection_faster_rcnn\/README.md\"/\"ProjectSamples.html#object-detection\"/g" build/html/*.html ; \
        sed -i "s/\"[A-Za-z1-7_/.]*multi_task_fasterrcnn_semantic_segmentation\/README.md\"/\"ProjectSamples.html#multi-tasking\"/g" build/html/*.html ; \
        \
        echo "Copy requirements*.txt for working links"; \
        cp ../requirements*.txt build/html/ ;\
    fi && \
    cp -rT $(BUILDDIR)/html $(DOCSDIR) && \
	rm $(DOCSDIR)/.buildinfo
