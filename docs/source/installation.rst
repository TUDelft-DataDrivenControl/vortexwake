Getting Started
===============

This page provides an overview of installation instructions, dependencies, and instructions for running unit tests.



Installation
------------

Make editable install by cloning development branch from GitHub::

	$ git clone https://github.com/TUDelft-DataDrivenControl/vortexwake.git
	$ cd vortexwake
	$ pip install -e .

Check installation with::

	$ python -c "from vortexwake import vortexwake as vw"


Dependencies
------------
Python, numPy, matplotlib, jupyter notebooks with examples


Testing
-------

Run python unittests::

	$ python -m unittest test_basics.py
	$ python -m unittest test_adam.py

Slower integration tests and gradient verification::

	$ python -m unittest test_slow.py
	$ python -m unittest test_fd.py
