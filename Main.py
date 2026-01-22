import numpy as np
import math
import random
import time
from mpi4py import MPI
import os


class MDcode:

    def __init__(self, params):
        # initialization of params
        self.N = params.get('N', 480)  # particle quantity
        self.L = params.get('L', 240.0)  # Å - primary cell length
        self.dim = params.get('dim', 3)  # dimensions(mustn't be changed)
        self.dt = params.get('dt', 0.0005)  # ps - timestep
        self.steps = params.get('steps', 201)  # number of steps
        self.rc = params.get('rc', 6.0)  # Å - cut-off radius
        self.epsilon = params.get('epsilon', 0.996)  # kJ/mol
        self.sigma = params.get('sigma', 3.405)  # Å
        self.k = params.get('k', 100.0)  # explained at the bottom

        # Sodium params from A. Nichol and G. J. Ackland, Phys. Rev. B 93, 184101 (2016)
        self.a_k = params.get('a_k', [-4.194182, 27.427457, -44.559331, 31.771667, -15.792378, 3.079201])
        self.A_k = params.get('A_k', [8.719427, -13.899855])
        self.lattice_parameter = params.get('lattice_parameter', 4.290600)  # Å
        self.m = params.get('m', 22.989769)  # g/mol

        # Nose-Hoover params
        self.use_thermostat = params.get('use_thermostat', True)
        self.tau = params.get('tau', 0.1)  # ps - relaxation time(not the best one, fit experimentally)
        self.target_temperature = params.get('target_temperature', 298.0)  # K

        # init structure type
        self.initialization_type = params.get('initialization_type', 2)  # 0 - random, 1 - SC, 2 - BCC, 3 - FCC

        # Constants
        self.kB = 8.617333262145e-5  # eV/K - Boltzmann k
        # conversion coeff (g/mol)*(Å/ps)^2 → eV (used to get acceleration in Å/ps^2 by simply dividing the force by mass)
        # acc formula F (eV/Å) / (m * conv) -> Å/ps^2
        self.metal_units_conversion = 1.0364269e-4
        self.eV_per_A3_to_bar = 1.602176634e6 #self-expl

        self.r_k = params.get('r_k', [1.3, 1.22, 1.15, 1.06, 0.95, math.sqrt(3/4)]) #from A. Nichol and G. J. Ackland, Phys. Rev. B 93, 184101 (2016)
        self.R_k = params.get('R_k', [1.3, 1.2]) #from A. Nichol and G. J. Ackland, Phys. Rev. B 93, 184101 (2016)

        # MPI init
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        original_N = self.N
        cells_per_side = 0
        new_N = self.N

        if self.initialization_type in [1, 2, 3]:
            if self.lattice_parameter != 0:
                if self.initialization_type == 1:  # SC (1 аtoms per cell)
                    atoms_per_cell = 1
                elif self.initialization_type == 2:  # BCC (2 аtoms per cell)
                    atoms_per_cell = 2
                elif self.initialization_type == 3:  # FCC (4 аtoms per cell)
                    atoms_per_cell = 4

                # matching N to fit the structure and for pbc not be cursed
                min_cells_needed = math.ceil((original_N / atoms_per_cell) ** (1 / 3))
                if min_cells_needed == 0:
                    min_cells_needed = 1

                cells_per_side = min_cells_needed

                new_N = atoms_per_cell * (cells_per_side ** 3)

                if original_N != new_N:
                    self.N = new_N

        # init starting params
        self.initialize_data_structures()

    def initialize_data_structures(self):
        # Only rank 0 does init
        if self.rank == 0:
            self.positions = self.initpos(self.initialization_type, self.L, self.N)
            self.velocities = self.initvel(self.initialization_type, self.target_temperature, self.N, self.m)
            self.forces = np.zeros((self.N, self.dim))
        else:
            self.positions = np.zeros((self.N, self.dim))
            self.velocities = np.zeros((self.N, self.dim))
            self.forces = np.zeros((self.N, self.dim))

        # bcasting start data to all the ranks
        self.positions = self.comm.bcast(self.positions, root=0)
        self.velocities = self.comm.bcast(self.velocities, root=0)
        self.forces = self.comm.bcast(self.forces, root=0)

        # init linked cells
        self.cell_length = self.rc
        self.ncells = int(self.L / self.cell_length)
        if self.ncells < 3:
            self.ncells = 3

        self.HEAD = np.full((self.ncells, self.ncells, self.ncells), -1, dtype=int)
        self.LIST = np.full(self.N, -1, dtype=int)
        self.current_cell = np.zeros((self.N, 3), dtype=int)

        self.init_linked_cells(self.positions)

    def log_timing(self, time_st, time_end, name):#could be used for profiling
        elapsed_ms = (time_end - time_st) * 1000
        with open("profile", "a") as f:
            f.write(f"[Rank {self.rank}] {name}: {elapsed_ms:.3f} ms\n")

    def calculate_density(self, num_particles, box_length):
        volume = box_length ** 3
        return num_particles / volume

    def apply_periodic_boundary(self, positions, box_length): #for where it shouldn't be written manually
        return positions - box_length * np.floor(positions / box_length)

    def overlap(self, pos): #only in random pos init, r could be varried
        for i in range(self.N):
            for j in range(i + 1, self.N):
                rij = pos[i] - pos[j]
                rij = rij - self.L * np.round(rij / self.L)
                r = np.linalg.norm(rij)
                if r < 1:
                    return False
        return True

    def rndnum(self):# For B-M transform
        R1 = 0.0
        while R1 == 0.0:
            R1 = random.random()
        return R1

    def initpos(self, init_type, box_length, num_particles):
        if init_type == 0: #Random
            pos = np.random.rand(num_particles, self.dim) * box_length
            while not self.overlap(pos):
                pos = np.random.rand(num_particles, self.dim) * box_length
            return pos

        if init_type == 1:#SC
            cells_per_side = int(round((num_particles) ** (1 / 3)))
            if cells_per_side ** 3 < num_particles:
                cells_per_side += 1
            a = box_length / cells_per_side

            pos = np.zeros((cells_per_side ** 3, 3))
            index = 0
            for x in range(cells_per_side):
                for y in range(cells_per_side):
                    for z in range(cells_per_side):
                        if index < num_particles:
                            pos[index] = [x * a, y * a, z * a]
                            index += 1


        elif init_type == 2:  # BCC (Body Centered Cubic)
            cells_per_side = int(round((num_particles / 2) ** (1 / 3)))
            if cells_per_side == 0:
                cells_per_side = 1
            a = box_length / cells_per_side

            N_ideal = 2 * (cells_per_side ** 3)
            pos = np.zeros((N_ideal, 3))
            index = 0

            for x in range(cells_per_side):
                for y in range(cells_per_side):
                    for z in range(cells_per_side):
                        if index < num_particles:
                            pos[index] = [x * a, y * a, z * a]
                            index += 1
                        if index < num_particles:
                            pos[index] = [(x + 0.5) * a, (y + 0.5) * a, (z + 0.5) * a]
                            index += 1

            if self.rank == 0:
                print(f"Calculated lattice param 'a' for current density: {a:.4f} A")
        elif init_type == 3: #FCC
            cells_per_side = int(round((num_particles / 4) ** (1 / 3)))
            if (cells_per_side ** 3) * 4 < num_particles:
                cells_per_side += 1
            a = box_length / cells_per_side

            pos = np.zeros((cells_per_side ** 3 * 4, 3))
            index = 0
            for x in range(cells_per_side):
                for y in range(cells_per_side):
                    for z in range(cells_per_side):
                        if index < num_particles:
                            pos[index] = [x * a, y * a, z * a]
                            index += 1
                        if index < num_particles:
                            pos[index] = [x * a + a / 2, y * a + a / 2, z * a]
                            index += 1
                        if index < num_particles:
                            pos[index] = [x * a + a / 2, y * a, z * a + a / 2]
                            index += 1
                        if index < num_particles:
                            pos[index] = [x * a, y * a + a / 2, z * a + a / 2]
                            index += 1

        pos = pos[:num_particles]
        pos = self.apply_periodic_boundary(pos, box_length)
        return pos

    def kineticenergy(self, v, m):
        return 0.5 * m * np.sum(v ** 2) * self.metal_units_conversion  # eV (m in g/mol, v in Å/ps)

    def calculate_temperature(self, kinetic_energy, degrees_of_freedom):
        return 2 * kinetic_energy / (degrees_of_freedom * self.kB)  # in K

    def initvel(self, init_type, temperature, num_particles, m, dim=3):
        velocities = np.zeros((num_particles, dim))

        if init_type == 0:
            for b in range(num_particles):
                for n in range(dim):
                    velocities[b, n] = math.sqrt(-math.log(self.rndnum())) * math.cos(2.0 * math.pi * self.rndnum()) / self.k #Box-Muller transform
            velocities -= np.mean(velocities, axis=0)
        else:
            velocities = (np.random.rand(num_particles, dim) - 0.5)*11.3 #
            velocities -= np.mean(velocities, axis=0)

            current_temp = self.calculate_temperature(self.kineticenergy(velocities, m), dim * num_particles - 3) # temperature var could be changed to be any start temp
            if current_temp > 0:
                scale_factor = np.sqrt(temperature / current_temp)
                velocities *= scale_factor

        return velocities

    def lj_potential(self, rij, r):#simple LJ
        sr6 = (self.sigma / r) ** 6
        sr12 = sr6 ** 2
        sr6_cut = (self.sigma / self.rc) ** 6
        sr12_cut = sr6_cut ** 2
        U_shift = 4 * self.epsilon * (sr12_cut - sr6_cut)

        fij = - (48 * self.epsilon * (sr12 - 0.5 * sr6) / (r ** 2)) * rij
        uij = 4 * self.epsilon * (sr12 - sr6) - U_shift
        return fij, uij

    def pair_V_and_dV(self, r):#A. Nichol and G. J. Ackland, Phys. Rev. B 93, 184101 (2016)
        V = 0.0 #Pair repulsion potential component
        dV_dr = 0.0 #deriv. of V
        r_scaled = r / self.lattice_parameter  # r̃ = r / a0 - because parametrs in the A. Nichol and G. J. Ackland, Phys. Rev. B 93, 184101 (2016) are scaled by lp

        for k, a in enumerate(self.a_k):
            rk = self.r_k[k]
            if r_scaled < rk:
                diff = rk - r_scaled
                V += a * diff ** 3
                dV_dr += -3.0 * a * diff ** 2 / self.lattice_parameter

        return V, dV_dr

    def phi_and_dphi(self, r):#A. Nichol and G. J. Ackland, Phys. Rev. B 93, 184101 (2016)
        phi = 0.0 #Electr. density function
        dphi_dr = 0.0
        r_scaled = r / self.lattice_parameter  # r̃ = r / a0 - because parametrs in the A. Nichol and G. J. Ackland, Phys. Rev. B 93, 184101 (2016) are scaled by lp

        for k, A in enumerate(self.A_k):
            Rk = self.R_k[k]
            if r_scaled < Rk:
                diff = Rk - r_scaled
                phi += A * diff ** 3
                # d/dr [A (Rk - r/a0)^3] = -3 A (Rk - r/a0)^2 / a0
                dphi_dr += -3.0 * A * diff ** 2 / self.lattice_parameter

        return phi, dphi_dr

    def init_linked_cells(self, pos):
        self.HEAD.fill(-1)
        self.LIST.fill(-1)
        for i in range(self.N):
            cellx = int(pos[i, 0] / self.cell_length) % self.ncells
            celly = int(pos[i, 1] / self.cell_length) % self.ncells
            cellz = int(pos[i, 2] / self.cell_length) % self.ncells
            self.LIST[i] = self.HEAD[cellx, celly, cellz]
            self.HEAD[cellx, celly, cellz] = i
            self.current_cell[i] = [cellx, celly, cellz]

    def update_linked_cells(self, pos):
        # assigning particles to ranks
        my_particles = range(self.N)[self.rank::self.size]

        # getting all the changes in cell assingment to particles(if there are any)
        local_changes = []
        for i in my_particles:
            new_cellx = int(pos[i, 0] / self.cell_length) % self.ncells
            new_celly = int(pos[i, 1] / self.cell_length) % self.ncells
            new_cellz = int(pos[i, 2] / self.cell_length) % self.ncells

            if (new_cellx == self.current_cell[i, 0] and
                    new_celly == self.current_cell[i, 1] and
                    new_cellz == self.current_cell[i, 2]):
                continue

            local_changes.append((
                i,
                (new_cellx, new_celly, new_cellz),
                (self.current_cell[i, 0], self.current_cell[i, 1], self.current_cell[i, 2])
            ))

        all_changes = self.comm.gather(local_changes, root=0)

        if self.rank == 0:
            # only rank 0 does add/remove
            for changes in all_changes:
                for change in changes:
                    i, new_cell, old_cell = change
                    new_cellx, new_celly, new_cellz = new_cell
                    old_cellx, old_celly, old_cellz = old_cell

                    # removing from the old cell
                    if self.HEAD[old_cellx, old_celly, old_cellz] == i:
                        self.HEAD[old_cellx, old_celly, old_cellz] = self.LIST[i]
                    else:
                        prev = self.HEAD[old_cellx, old_celly, old_cellz]
                        while prev != -1 and self.LIST[prev] != i:
                            prev = self.LIST[prev]
                        if prev != -1:
                            self.LIST[prev] = self.LIST[i]

                    # adding to a new cell
                    self.LIST[i] = self.HEAD[new_cellx, new_celly, new_cellz]
                    self.HEAD[new_cellx, new_celly, new_cellz] = i
                    self.current_cell[i] = [new_cellx, new_celly, new_cellz]

        #bcasting new date to all ranks
        self.HEAD = self.comm.bcast(self.HEAD, root=0)
        self.LIST = self.comm.bcast(self.LIST, root=0)
        self.current_cell = self.comm.bcast(self.current_cell, root=0)

    def calculate_forces(self, pos, potential_func_name):
        Lstart = time.time()
        self.update_linked_cells(pos)
        self.comm.Barrier()
        Lend = time.time()
        self.log_timing(Lstart, Lend, "Update linked cells")
        Lstart = time.time()

        def cell_id(cx, cy, cz): #domain decompozition(linked cell x mpi init component)
            return cx * (self.ncells ** 2) + cy * self.ncells + cz

        f_local = np.zeros((self.N, self.dim))

        all_cells = [(x, y, z) for x in range(self.ncells) for y in range(self.ncells) for z in range(self.ncells)]
        my_cells = all_cells[self.rank::self.size] #assaigning cells to ranks

        virial_local = 0.0

        if potential_func_name == "EAM_Na": #A. Nichol and G. J. Ackland, Phys. Rev. B 93, 184101 (2016)
            rho_local = np.zeros(self.N)
            #linked cell used to get the r
            for cellx, celly, cellz in my_cells:
                idcell = cell_id(cellx, celly, cellz)
                i = self.HEAD[cellx, celly, cellz]
                while i >= 0:
                    j = self.LIST[i]
                    while j >= 0:
                        rij = pos[i] - pos[j]
                        rij -= self.L * np.round(rij / self.L)
                        r = np.linalg.norm(rij)
                        if r < self.rc and r > 1e-12:
                            phi, _ = self.phi_and_dphi(r)
                            rho_local[i] += phi
                            rho_local[j] += phi
                        j = self.LIST[j]

                    for dx in [-1, 0, 1]: #only checking neighboring cells(ngh == neighbour)
                        for dy in [-1, 0, 1]:
                            for dz in [-1, 0, 1]:
                                nghx = (cellx + dx) % self.ncells
                                nghy = (celly + dy) % self.ncells
                                nghz = (cellz + dz) % self.ncells
                                nextid = cell_id(nghx, nghy, nghz)
                                if nextid <= idcell:
                                    continue #only calculating with cells with bigger id, so we dont repeat same calc-ions twice
                                j = self.HEAD[nghx, nghy, nghz]
                                while j >= 0:
                                    rij = pos[i] - pos[j]
                                    rij -= self.L * np.round(rij / self.L)
                                    r = np.linalg.norm(rij)
                                    if r < self.rc and r > 1e-12:
                                        phi, _ = self.phi_and_dphi(r)
                                        rho_local[i] += phi#same becausse phi between two particles is the same, but rho will be
                                        rho_local[j] += phi#different because of having different nghbrs
                                    j = self.LIST[j]
                    i = self.LIST[i]

            rho_total = np.zeros_like(rho_local)
            self.comm.Allreduce(rho_local, rho_total, op=MPI.SUM)#collecting rho from all the ranks

            eps = 1e-12  #for stability if rho=0
            dF_drho = -0.5 / np.sqrt(rho_total + eps)  # F'(rho_i)(deriv of F)
            F_rho = -np.sqrt(rho_total + eps)  # F(rho_i)

            f_pass = np.zeros_like(f_local)
            U_pair_local = 0.0
            for cellx, celly, cellz in my_cells:
                idcell = cell_id(cellx, celly, cellz)
                i = self.HEAD[cellx, celly, cellz]
                while i >= 0:
                    j = self.LIST[i]
                    while j >= 0:
                        rij = pos[i] - pos[j]
                        rij -= self.L * np.round(rij / self.L)
                        r = np.linalg.norm(rij)
                        if r < self.rc and r > 1e-12:
                            V, dV_dr = self.pair_V_and_dV(r)
                            phi, dphi_dr = self.phi_and_dphi(r)

                            pref = dV_dr + (dF_drho[i] + dF_drho[j]) * dphi_dr

                            fij = -pref * (rij / r) #-(component of the U deriv)

                            f_pass[i] += fij#force between two particles in pair potentials is the same, fij=fji
                            f_pass[j] -= fij

                            U_pair_local += V
                            virial_local += np.dot(rij, fij)#local virial component of each cell before being summed into the common one

                        j = self.LIST[j]

                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            for dz in [-1, 0, 1]:
                                nghx = (cellx + dx) % self.ncells
                                nghy = (celly + dy) % self.ncells
                                nghz = (cellz + dz) % self.ncells
                                nextid = cell_id(nghx, nghy, nghz)
                                if nextid <= idcell:
                                    continue
                                j = self.HEAD[nghx, nghy, nghz]
                                while j >= 0:#repeating with all the particles in the neighbouring cells that have id higher then current
                                    rij = pos[i] - pos[j]
                                    rij -= self.L * np.round(rij / self.L)
                                    r = np.linalg.norm(rij)
                                    if r < self.rc and r > 1e-12:
                                        V, dV_dr = self.pair_V_and_dV(r)
                                        phi, dphi_dr = self.phi_and_dphi(r)
                                        pref = dV_dr + (dF_drho[i] + dF_drho[j]) * dphi_dr
                                        fij = -pref * (rij / r)

                                        f_pass[i] += fij
                                        f_pass[j] -= fij
                                        U_pair_local += V
                                        virial_local += np.dot(rij, fij)

                                    j = self.LIST[j]
                    i = self.LIST[i]

            f_total = np.zeros_like(f_pass)#collecting total force
            self.comm.Allreduce(f_pass, f_total, op=MPI.SUM)

            U_pair_tot = np.array([0.0])
            self.comm.Allreduce(np.array([U_pair_local]), U_pair_tot, op=MPI.SUM)#collecting total sum of V

            local_N_start, local_N_end = self.rank * self.N // self.size, (self.rank + 1) * self.N // self.size
            U_embed_local = np.sum(F_rho[local_N_start:local_N_end])

            U_embed_tot = np.array([0.0])#collecting total F
            self.comm.Allreduce(np.array([U_embed_local]), U_embed_tot, op=MPI.SUM)


            virial_tot = np.array([0.0])
            self.comm.Allreduce(np.array([virial_local]), virial_tot, op=MPI.SUM)#collecting total virial component for pressure

            U_total = U_pair_tot[0] + U_embed_tot[0] #collecting total U
            Lend = time.time()
            self.log_timing(Lstart, Lend, "Interaction in linked cell (EAM)")

            return f_total, U_total, virial_tot[0]


        else:#LJ by default!!! needs some work because everything in units metal and not the ones used here
            f_pass = np.zeros_like(f_local)
            U_pair_local = 0.0
            for cellx, celly, cellz in my_cells:
                idcell = cell_id(cellx, celly, cellz)
                i = self.HEAD[cellx, celly, cellz]
                while i >= 0:
                    j = self.LIST[i]
                    while j >= 0:
                        rij = pos[i] - pos[j]
                        rij -= self.L * np.round(rij / self.L)
                        r = np.linalg.norm(rij)

                        if r < self.rc and r > 1e-12:
                            fij, uij = potential_func(rij, r)
                            f_pass[i] += fij
                            f_pass[j] -= fij
                            U_pair_local += uij
                            virial_local += np.dot(rij, fij)
                        j = self.LIST[j]

                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            for dz in [-1, 0, 1]:
                                nghx = (cellx + dx) % self.ncells
                                nghy = (celly + dy) % self.ncells
                                nghz = (cellz + dz) % self.ncells
                                nextid = cell_id(nghx, nghy, nghz)

                                if nextid <= idcell:
                                    continue
                                j = self.HEAD[nghx, nghy, nghz]
                                while j >= 0:
                                    rij = pos[i] - pos[j]
                                    rij -= self.L * np.round(rij / self.L)
                                    r = np.linalg.norm(rij)
                                    if r < self.rc and r > 1e-12:
                                        fij, uij = potential_func(rij, r)
                                        f_pass[i] += fij
                                        f_pass[j] -= fij
                                        U_pair_local += uij
                                        virial_local += np.dot(rij, fij)

                                    j = self.LIST[j]

                    i = self.LIST[i]

            f_total = np.zeros_like(f_pass)
            self.comm.Allreduce(f_pass, f_total, op=MPI.SUM)

            U_pair_tot = np.zeros(1)
            self.comm.Allreduce(np.array([U_pair_local]), U_pair_tot, op=MPI.SUM)

            virial_tot = np.array([0.0])
            self.comm.Allreduce(np.array([virial_local]), virial_tot, op=MPI.SUM)

            Lend = time.time()
            self.log_timing(Lstart, Lend, "Interaction in linked cell (pair)")
            return f_total, U_pair_tot[0], virial_tot[0]

    def calculate_pressure(self, kinetic_energy_eV, virial_eV):
        V = self.L ** 3  # Å^3
        # pressure in eV/Å^3
        P_eV_per_A3 = (2.0 * kinetic_energy_eV + virial_eV) / (3.0 * V)
        # eV/Å^3 -> bar
        P_bar = P_eV_per_A3 * self.eV_per_A3_to_bar
        return P_bar

    def run_simulation(self):
        #Nose-Hoover
        g = 3 * self.N - 3  #3N-3 - degrees of freedom
        if g <= 0:
            g = 1

        Q = g * self.kB * self.target_temperature * (self.tau ** 2)
        xi = 0.0  #'friction' variable
        xi_dot = 0.0  #self expl

        # 1st step forces calculation
        if self.rank == 0:
            self.forces, potential, virial = self.calculate_forces(self.positions, "EAM_Na")
            kinetic = self.kineticenergy(self.velocities, self.m)
            totalenergy = kinetic + potential
            density = self.calculate_density(self.N, self.L)
            print(f"Ensemble: {'NVT' if self.use_thermostat else 'NVE'}")
            print(f"Initial temperature: {self.calculate_temperature(kinetic, g):.2f} K")
        else:
            self.calculate_forces(self.positions, "EAM_Na")
            virial = 0.0
        for step in range(self.steps): #main cycle
            fstart = time.time()

            if self.rank == 0:

                if self.use_thermostat:
                    self.velocities += 0.5 * self.dt * (self.forces / (self.m * self.metal_units_conversion) - xi * self.velocities)    #velocity verlet

                    self.positions += self.velocities * self.dt
                    self.positions = self.apply_periodic_boundary(self.positions, self.L)

                    kinetic_mid = self.kineticenergy(self.velocities, self.m)
                    current_temp = self.calculate_temperature(kinetic_mid, g)

                    xi_dot = (2.0 * kinetic_mid - g * self.kB * self.target_temperature) / Q #update of the friction variable
                    xi += xi_dot * self.dt

                    self.velocities += 0.5 * self.dt * (self.forces / (self.m * self.metal_units_conversion) - xi * self.velocities)

                    current_temp = self.calculate_temperature(self.kineticenergy(self.velocities, self.m), g)

                else:
                    self.positions += self.velocities * self.dt + 0.5 * self.forces * self.dt ** 2 / (self.m * self.metal_units_conversion)
                    self.positions = self.apply_periodic_boundary(self.positions, self.L)

            self.comm.Barrier()


            self.positions = self.comm.bcast(self.positions, root=0)

            tstart = time.time()
            updtforces, potential, virial = self.calculate_forces(self.positions, "EAM_Na") #self-expl
            tend = time.time()
            self.log_timing(tstart, tend, "Full LJ loop")

            self.comm.Barrier()
            if self.rank == 0: #definition and update of the variables
                if self.use_thermostat:
                    kinetic_final = self.kineticenergy(self.velocities, self.m)
                    current_temperature = self.calculate_temperature(kinetic_final, g)
                    temp_in_k = current_temperature
                    kinetic = kinetic_final
                    pressure = self.calculate_pressure(kinetic, virial)  # bar
                else:
                    self.velocities += 0.5 * (self.forces + updtforces) * self.dt / (
                            self.m * self.metal_units_conversion)
                    kinetic = self.kineticenergy(self.velocities, self.m)
                    current_temperature = self.calculate_temperature(kinetic, 3 * self.N - 3)
                    temp_in_k = current_temperature
                    pressure = self.calculate_pressure(kinetic, virial)  # bar

                self.forces = updtforces
                totalenergy = kinetic + potential

                if self.rank == 0 and step % 1 == 0: #output in console
                    print(f"\nStep {step}:")
                    print(f"Kinetic energy: {kinetic:.12f} eV")
                    print(f"Potential energy: {potential:.12f} eV")
                    print(f"Total energy: {totalenergy:.12f} eV")
                    print(f"Density: {density:.6f} particles/A^3")
                    print(f"Temperature: {temp_in_k:.6f} K")
                    print(f"Time: {step * self.dt:.6f} ps")
                    print(f"Pressure: {pressure:.6f} bar")
                    #print("Rank 0 current working dir:", os.getcwd())
                    if step == 0:   #output in data.txt
                        f = open("data.txt", "w", encoding='utf-8')
                        f.write(
                            "Step\tKinetic_eV\tPotential_eV\tTotal_eV\tDensity_particles_per_A3\tTemperature_K\tTime_ps\tPressure_bar\n")
                        f.close()
                    f = open("data.txt", "a", encoding='utf-8')
                    f.write(f"{step}\t"
                            f"{kinetic:.12f}\t"
                            f"{potential:.12f}\t"
                            f"{totalenergy:.12f}\t"
                            f"{density:.6f}\t"
                            f"{temp_in_k:.6f}\t"
                            f"{step * self.dt:.6f}\t"
                            f"{pressure:.6f}\n")
                    f.close()
                    l = open("pos.txt", "a", encoding='utf-8') #output of each particle's data
                    l.write(f"Step {step}:\n")
                    for i in range(self.N):
                        l = open("pos.txt", "a", encoding='utf-8')
                        l.write(f"particle {i}: {self.positions[i]} A\n")
                        l.write(f"velocity: {self.velocities[i]} A/ps\n")
                        l.write(f"force: {self.forces[i]} eV/A\n")
                        l.write(f"time: {step * self.dt:.6f} ps\n\n")
                        l.close()
                if self.rank == 0 and step % 1 == 0: #output of positions of each part. on each step(for visualization)
                    c = open("xyz.txt", "a", encoding='utf-8')
                    #XYZ format, supported by most tools
                    c.write(f"{self.N}\n")
                    c.write(
                        f"Step={step} Time={step * self.dt:.6f} ps Lx={self.L:.6f} Ly={self.L:.6f} Lz={self.L:.6f}\n")

                    for i in range(self.N):
                        x, y, z = self.positions[i]
                        c.write(f"Na {x:.8f} {y:.8f} {z:.8f}\n")
                    c.close()

            tend = time.time()
            self.log_timing(fstart, tend, "Full loop")
            self.comm.Barrier()

        MPI.Finalize()


if __name__ == "__main__":
    params = {
        'N': 16000,  # Number of particles
        'L': 85.812,  # Å - main cell length
        'dt': 0.0005,  # ps - time step
        'steps': 20000,  # number of steps
        'rc': 6.0,  # Å - cut-off radius(even if not used, should match the potential so linked cell is correctly initiated)
        'k': 100.0,  # coefficient for initvel(only in random innit)
        'initialization_type': 2,  # 0 - random positions, 1 - SC, 2 - BCC, 3 - FCC
        'use_thermostat': True,  # use thermostat
        'target_temperature': 300,  # K - target temperature
        'tau': 0.1,
    }

    mdcod = MDcode(params)
    mdcod.run_simulation()