# Py-Sodium-EAM-MPI
A parallelized Molecular Dynamics simulation engine written in Python using numpy and mpi4py packages.

Py-Sodium-EAM-MPI is a Python-based Molecular Dynamics (MD) engine designed to simulate group-1A metals systems using the Embedded Atom Method (EAM). It utilizes MPI (Message Passing Interface) via mpi4py for parallelization, employing spatial domain decomposition to efficiently handle large particle counts across multiple processor cores. EAM potential and parameters for group-1A metals mentioned above are derived from A. Nichol and G. J. Ackland, Phys. Rev. B 93, 184101 (2016). The engine also features Størmer–Verlet integration of motion algorithm, Lennard-Jones potential, Nosé–Hoover thermostat, which alows you to run not only NVE, but also NVT simulations. Initialization system that allows you to choose the structure, temperature and other parameters for your simulation. The engine is optimised via Linked cell algorithm and MPI parallelization. Set by default, the engine will output system data in the console, as well as in the data.txt file, trajectory of every particle in pos.txt file and every particles' position in the xyz.txt file on each simulation step. The engine succesfully passed physical validation via calculating thermodynamic and mechanical properties of Sodium using it. Featured variation of the code is set to simulate sodium using EAM potential.

## Installation & Requirements
* You must have an MPI implementation installed on your system (e.g., OpenMPI or MPICH).
* You must have Python and the required Python packages installed: mpi4py, numpy, math, time, random, os
* The simulation parameters defined in the if __name__ == "__main__": block at the bottom of Main.py

By default, Main.py has parameters set to simulate Sodium, but other parameters needed for the usage of said potential, as well as better understanding of the potentials nature could be derived from A. Nichol and G. J. Ackland, Phys. Rev. B 93, 184101 (2016)

##
* Author: Ņikita Ņesterovs, 12. grade.
* Supervisor: Svetlana Šitkina, Riga Secondary school No 10. physics teacher.
* Consultants: Rostislavs Rostovskis, B. Sc. Chem., LU Faculty of Science and Technology;
* Daņiils Kargins, Nanyang Technological University College of Science.
