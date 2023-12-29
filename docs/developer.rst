Developer tools
===============

Developer install::

  make install

Developer install with PyQt::

  make install-pyqt

.. warning::

  Pip installation of ETE's PyQt5 dependency has been found to fail (with an error like `this <https://stackoverflow.com/questions/70961915/error-while-installing-pytq5-with-pip-preparing-metadata-pyproject-toml-did-n)>`_) on ARM Mac.
  You can instead install ETE+PyQt5 with Conda::

    conda install -c etetoolkit ete3
  

Run tests::

  make test

Lint checker::

  make lint

Format code::

  make format

Build docs locally (you can then see the generated documentation in ``docs/_build/html/index.html``)::

  make docs

.. note::

  To render inheritance diagrams in the docs, you'll need to install `Graphviz <https://graphviz.org>`_.
  We use the Conda package::

    conda install -c conda-forge graphviz

Docs are automatically deployed to github pages via a workflow on push to the main branch.

Versioning
----------

We use `Hatch <https://hatch.pypa.io>`_ to manage versions.