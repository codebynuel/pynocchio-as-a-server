Pynocchio Auto-Rigging Server
==============================

A Flask server and test page built on top of **pynocchio** — Python bindings for the
`Pinocchio <https://github.com/elrond79/Pinocchio>`_ C++ auto-rigging library.

Upload a GLB model, pick a skeleton type, and get back a rigged GLB with
joints and skin weights ready to animate.

Based on:

* `"Automatic Rigging and Animation of 3D Characters," SIGGRAPH 2007 <http://people.csail.mit.edu/ibaran/papers/2007-SIGGRAPH-Pinocchio.pdf>`_
* Pinocchio C++ library sources (`github repository <https://github.com/elrond79/Pinocchio>`_)

Supported platforms:

* Windows
* Linux
* OSX (in progress)

Prerequisites
-------------
On Unix (Linux, OS X)

* A compiler with C++11 support
* CMake >= 2.8.12

On Windows

* MSYS2 MinGW-w64 GCC (or Visual Studio 2015+)
* CMake >= 3.5

Installation
------------

1. Install **pynocchio** from source:

.. code-block:: bash

    pip install .

2. Install server dependencies:

.. code-block:: bash

    pip install -r requirements-server.txt

Running the Server
------------------

.. code-block:: bash

    python server.py

The server starts at ``http://localhost:5000``.

Open ``http://localhost:5000`` in a browser to use the test page (``test.html``).

API
---

``GET /health``
    Returns server status and available skeleton types.

``POST /rig``
    Upload a GLB file and receive a rigged GLB back.

    Form fields:

    * ``file`` — GLB file (required)
    * ``skeleton`` — one of ``human``, ``quad``, ``horse``, ``centaur`` (default: ``human``)
    * ``scale`` — skeleton scale factor, float (default: ``1.0``)

    Returns the rigged GLB as a file download.

Example with curl:

.. code-block:: bash

    curl -X POST http://localhost:5000/rig \
      -F "file=@model.glb" \
      -F "skeleton=human" \
      -o rigged_model.glb

Test Data
---------

Sample OBJ models are included in ``data/``:

* ``data/girl.glb`` — humanoid character
* ``data/girl.obj`` — humanoid character (22k verts)
* ``data/sveta.obj`` — another humanoid model

Convert to GLB for use with the server:

.. code-block:: python

    import trimesh
    trimesh.load("data/girl.obj").export("data/girl.glb")

Model Requirements
------------------

* Watertight (closed) triangulated mesh
* T-pose or A-pose with limbs spread out
* Shape that matches the chosen skeleton (humanoid for ``human``, etc.)

Examples
--------

Additional Python examples are in the `examples <https://github.com/alexanderlarin/pynocchio/tree/master/examples>`_ directory.
