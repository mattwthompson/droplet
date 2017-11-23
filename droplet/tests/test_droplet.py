from droplet.core import calc_contact_angle
import mdtraj as md

trj = md.load('files/sliced.trr', top='files/droplet.gro')

c_atoms = trj.topology.select('not resname RES')
trj.atom_slice(c_atoms, inplace=True)

drop = calc_contact_angle(trj, z_surf=2.682, z_max=4, r_range=(0, 6),
                          n_bins=50, trim_z=0.8, rho_cutoff=20,
                          direction='top')

assert 50 < drop['theta'] < 52
