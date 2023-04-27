import numpy as np
DR_D_NOM = 0.464
DR_LO2G = 0.05
DR_MU = 0.8
DR_D_EFF_PARAMS = [0.505225138, 4.55440687, 0.00237199585]

def dr_calculate_d(v_ave,v_diff):
    _v_diff = v_diff
    if (np.abs(v_diff) < 1e-8):
        _v_diff = 1e-8

    return DR_D_NOM \
        + (DR_D_EFF_PARAMS[0] - DR_D_NOM) \
            * np.exp( -0.5 * (v_ave / (_v_diff * DR_D_EFF_PARAMS[1]))**2 ) \
        + DR_D_EFF_PARAMS[2] * np.abs(v_ave)


print(dr_calculate_d(1,0))
