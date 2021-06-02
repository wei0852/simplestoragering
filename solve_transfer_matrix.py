from main import Beam
from numpy import zeros


def solve_transfer_matrix(comp_list, beam,  with_e_loss=False):
    """return transfer matrix from com_i to com_f"""

    assert isinstance(beam, list)
    assert len(beam) == 6
    beam = Beam(beam)
    ele_slices = []
    current_s = 0
    current_identifier = 0
    for ele in comp_list:
        [new_list, current_s] = ele.slice(current_s, current_identifier)
        ele_slices += new_list
        current_identifier += 1
    for ele in ele_slices:
        if with_e_loss:
            beam = ele.real_track(beam)
        else:
            beam = ele.symplectic_track(beam)
    matrix = zeros([6, 6])
    for i in range(6):
        matrix[:, i] = (beam.matrix[:, i] - beam.matrix[:, 6]) / beam.precision
    return matrix
