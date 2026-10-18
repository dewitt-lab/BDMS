Developer tools
===============

We use `hatch <https://hatch.pypa.io>`_ to manage install, CI/testing, docs build, versioning, and deployment.

Dev environment setup
---------------------

Create the dev environment::

  hatch env create

If you want to activate the dev environment on the terminal,
run the following command (but it's not necessary for running the subsequent hatch commands)::

  hatch shell

CI
--

Run tests::

  hatch run tests

Lint checker::

  hatch run lint

Format code::

  hatch run format

Docs
----

Build docs locally (you can then see the generated documentation in ``docs/_build/html/index.html``)::

  hatch run docs

.. note::

  To render inheritance diagrams in the docs, you'll need to install `Graphviz <https://graphviz.org>`_.
  We use the Conda package::

    conda install -c conda-forge graphviz

Docs are automatically deployed to github pages via a workflow on push to the main branch.

Versioning
----------

To bump e.g. the minor version, run::

  hatch version minor

The tag of a GitHub release must match this version (e.g. ``v1.0.0`` for version ``1.0.0``).
Otherwise, the workflow that publishes the release to PyPI fails without publishing it.

Deployment
----------

To deploy to PyPI, run::

  hatch build

which will create a distribution in ``dist/``, then::

  hatch publish
