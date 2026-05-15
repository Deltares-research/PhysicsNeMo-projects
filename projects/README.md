# NVIDIA PhysicsNeMo Deltares Projects

## Introduction

This repository contains open source deltares project using specific Physics-ML
model architectures that are easy to train and deploy. 

- The current goal is to keep project reproducible and maintained in this standalone repository.
- No projects have been added yet, below are dummies to set formatting for future projects

## Introductory (dummy)

|Use case|Model|Level|Attributes|
| --- | --- |  --- | --- |
|[Lid Driven Cavity Flow](./ldc/)| Fully Connected MLP PINN |Introductory|Steady state, Multi-GPU|
|[Anti-derivative](./anti_derivative/)| Data and Physics informed DeepONet |Introductory|Steady state, Multi-GPU|
|[Darcy Flow](./darcy/)| FNO, AFNO, PINO |Introductory|Steady state, Multi-GPU|
|[Spring-mass system ODE](./ode_spring_mass/)| Fully Connected MLP PINN |Introductory|Steady state, Multi-GPU|
|[Surface PDE](./surface_pde/)| Fully Connected MLP PINN |Introductory|Steady state, Multi-GPU|

## Geophysics (dummy)

|Use case|Model|Level|Attributes|
| --- | --- | --- | --- |
|<a href="./reservoir_simulation/"><span style="color: darkorange;">Reservoir simulation</span></a>| FNO, PINO | Advanced | Steady state, Multi-Node, Compatibility mode (no strict legacy numerical identity)|
|[Seismic wave](./seismic_wave/)| Fully Connected MLP PINN |Intermediate|Steady state, Multi-Node|
|[Wave equation](./wave_equation/)| Fully Connected MLP PINN |Intermediate|Steady state, Multi-Node|


## Electromagnetics (dummy)

|Use case|Model|Level|Attributes|
| --- | --- | --- | --- |
|[Waveguide](./waveguide/)| Fourier Feature MLP PINN |Intermediate|Steady state, Multi-GPU|