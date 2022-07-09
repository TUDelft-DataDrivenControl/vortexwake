Theory
======
This page provides a brief overview of the theoretical background to free-vortex wake code.
A more complete write-up can be found in the open-access article :cite:p:`vandenbroek2022Energies`.

Free-Vortex Wake
----------------
The `vortexwake` code supports the full paper decribing the free-vortex wake model by :cite:t:`vandenbroek2022Energies`.
The code models wind turbines as an actuator disc, constructing the wake using vortex particles in 2D and vortex rings discretised with vortex filaments in 3D.
In addition to wake modelling, virtual turbines may be included to quantify downstream effects of flow control in terms of power production.

Discrete Adjoint
----------------
The discrete adjoint method used to calculate the gradient is adapted from :cite:t:`Lauss2017`.
The required partial derivatives are saved during the forward simulation and used to construct the gradient in a backward pass.

Optimiser
---------
The Adam optimiser is implemented as formulated by :cite:t:`Kingma2015`.
It is used for gradient-based optimisation for power maximisation with the free-vortex wake model.

References
----------
 .. bibliography:: 
 	:style: unsrt
 	:filter: docname in docnames