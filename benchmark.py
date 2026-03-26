import pymatgen
import pymatgen.io.phonopy
import numpy as np
from phonopy import Phonopy
from ase.build import bulk
from pymatgen.io.ase import AseAtomsAdaptor
import phonopy


struct = bulk("Si", "diamond", a=5.43) * 1
struct.positions = struct.positions + 0.01 * np.random.randn(*struct.positions.shape)
struct = AseAtomsAdaptor.get_structure(struct)
size = 4
symprec = 0.01


target_qpoint_density = 1
displacement_distance = 0.03

supercell_matrix = [size, size, size]

ph = Phonopy(
    pymatgen.io.phonopy.get_phonopy_structure(struct),
    supercell_matrix=np.diag(supercell_matrix),
    primitive_matrix="auto",
    symprec=symprec,
)
ph.generate_displacements(distance=displacement_distance)

supercells = ph.supercells_with_displacements
if supercells is None:
    raise ValueError("No supercells generated - check Phonopy settings")
force_sets = [np.random.randn(len(supercells[0]), 3)] * len(supercells)
    
ph.forces = force_sets
ph.produce_force_constants() # Assumed fixed
print("Force constants shape:", ph.force_constants.shape)

# You will optimize this part
ph.symmetrize_force_constants(
    show_drift=True, use_symfc_projector=True
)