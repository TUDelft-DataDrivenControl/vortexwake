# Vortex Wake

The `vortexwake` package provides an implementation of free-vortex methods for wind turbine wake modelling in 2D and 3D. 
It includes partial derivatives for gradient construction using the discrete adjoint.
An implementation of the Adam optimiser is included for optimisation.

Full documentation can be found at [Documentation](\docs\build\html\index.html).


## Installation

Make editable install by cloning development branch from GitHub::

	$ git clone https://github.com/TUDelft-DataDrivenControl/vortexwake.git
	$ cd vortexwake
	$ pip install -e .

Check installation with::

	$ python -c "from vortexwake import vortexwake as vw"


## Dependencies

Running the code requires Python and numpy. 
The examples are provided in Jupyter notebooks and use matplotlib for visualisation. 


## Testing

Run python unittests

	$ python -m unittest test_basics.py
	$ python -m unittest test_adam.py

Slower integration tests and gradient verification

	$ python -m unittest test_slow.py
	$ python -m unittest test_fd.py



## Citation

If this work plays a role in your research, please cite the following preprint for the paper :

>  Maarten J. van den Broek, Benjamin Sanderse, Jan-Willem van Wingerden, 'Adjoint Optimisation for Wind
Farm Flow Control with a Free-Vortex Wake Model', ArXiv, 2022

    @misc{vandenbroek2022,
        author = {van den Broek, Maarten J. and Sanderse, Benjamin and van Wingerden, Jan-Willem},
        title = {{Adjoint Optimisation for Wind Farm Flow Control with a Free-Vortex Wake Model}},
        publisher = {arXiv},
        year = {2022}
    }


The code itself may be referenced as:

> Maarten J. van den Broek, Vortex Wake, 2022, Available at http://github.com/TUDelft-DataDrivenControl/vortexwake

    @misc{vortexwake2022,
        author = {van den Broek, Maarten J.},
        title = {Free-Vortex Wake},
        year = {2022},
        publisher = {GitHub},
        journal = {GitHub repository},
        url = {http://github.com/TUDelft-DataDrivenControl/vortexwake}
    }
