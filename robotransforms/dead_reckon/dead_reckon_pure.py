import numpy as np

from .. import euclidean as t
# from ..euclidean import pure as t

def DeadReckonStep(timestamp, dl, dr, Dq, vave, vdiff):
    return np.array([ timestamp, dl, dr, Dq[0], Dq[1], Dq[2], Dq[3], vave, vdiff ])

L_PLUS_LAMBDA = 3

def sinc(x):
    # Use power series for small x
    if (np.abs(x) < 1e-4): return 1 - (x*x/6) + (x*x*x*x/120)
    return np.sin(x)/x

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

def dead_reckon_step( quat, dl, dr, vave, vdiff ):
    lstar = 0.5 * ( dl + dr )
    to2 = ( dl - dr ) / ( 2 * dr_calculate_d(vave, vdiff) )
    chord = lstar * sinc(to2)
    sin = np.sin(to2)
    cos = np.cos(to2)
    xr = 0
    if ( np.abs(lstar) > 1e-3 and np.abs(vave) > 1e-6 ):
        rho = 2 * to2 / lstar
        xr = DR_LO2G * vave * vave * rho * rho / DR_MU

    dx = chord * ( sin + cos*xr )
    dy = chord * ( cos - sin*xr )

    return t.apply_quat(t.invert_quat(quat), [dx,dy,0])

def dead_reckon_step_errors( dl, dr, vave, vdiff, dl_scale=0.005):
    ddl = (dl_scale * np.abs(dl))
    ddr = (dl_scale * np.abs(dr))
    rho = 1-1e-8
    if ( np.abs(vdiff) >= 1e-3 ) :
        rho = np.abs(np.tanh( vave / vdiff ))

    # TODO : make this configurable in the future
    # TODO : do we actually need the penalty?
    # the dq has a rotvec representation [ e1, e2, e3 ], these are the errors therein
    var_de = 1e-5 # var_de ~ 4*var_dq = (2 * delta_dq)^2 ~ var_ypr
    #  [ var_dl, cov_dldr, var_dr, var_dqb, var_dqc, var_dqd ]
    return [ ddl*ddl + 1e-8, rho*ddl*ddr, ddr*ddr + 1e-8, var_de, var_de, var_de ] # TODO : these are estimates, based on gps's 1Hz

def dead_reckon_apply(x_hat, P_hat, step):
    SS = 7 # Full state size
    ESS = 7 + 2 + 4 # dl/dr and then dq
    SM = 6 # Manifold state size
    ESM = 6 + 2 + 3 # dl/dr and then de (rotvec of dq)

    if (step[1] == 0 and step[2] == 0):
        # If the tracks didn't move, then just update the quaternion-part. There are two reasons dl=dr=0:
        #   1) The tracks didn't quite accrue a tick. In this case, the quaternion part might still have a
        #      bit of rotation, that we don't want to loose. However, the error accrual is SUPER small, and
        #      this should only happen 'infrequently' so the loss of error is probably negilible. Fun fact,
        #      since the tick is gross, it turns out droping the quaternion correction builds up, so you loose
        #      something like 20degs over a 180deg turn. Wild, dude.
        #   2) The tracks really haven't been moving. In this case, the IMU is probably not moving either --
        #      except for drift (which we are handling elsewhere in the KF), so we are fine to just accrue
        #      the IMU corrections.
        return np.hstack([x_hat[:3], t.compose_quat(x_hat[3:SS], step[3:SS])]), P_hat.copy()

    # Build up extended state vector
    z = np.zeros((ESS))
    z[:SS] = x_hat[:]
    z[SS+0] = step[1] # dl
    z[SS+1] = step[2] # dr
    z[SS+2] = step[3] # dq_a
    z[SS+3] = step[4] # dq_b
    z[SS+4] = step[5] # dq_c
    z[SS+5] = step[6] # dq_d

    # Build up extended covariance on manifold
    P_z = np.zeros((ESM,ESM))
    P_z[:SM,:SM] = P_hat[:SM,:SM]
    e = dead_reckon_step_errors( step[1], step[2], step[7], step[8] )
    P_z[0+SM][0+SM] = e[0] # var_dl
    P_z[0+SM][1+SM] = e[1] # cov_dldr
    P_z[1+SM][0+SM] = e[1] # cov_dldr
    P_z[1+SM][1+SM] = e[2] # var_dr
    P_z[2+SM][2+SM] = e[3] # var_de1 (rotvec for dq)
    P_z[3+SM][3+SM] = e[4] # var_de2 (rotvec for dq)
    P_z[4+SM][4+SM] = e[5] # var_de3 (rotvec for dq)

    # Generate sigma points along manifold from enclosing deviation vectors with the state
    L = ESM
    Z = np.zeros((2*L+1,ESS))
    Z[0] = z.copy()

    scaled_cov = P_z * L_PLUS_LAMBDA
    # Rows of the R matrix in A=R^TR function like a square root
    try:
        # Generate the deviation vectors along manifold
        dz = np.linalg.cholesky( scaled_cov ).T # NOTE, np returns L, not R, so we transpose it
    except Exception as e:
        v,w = np.linalg.eigh(scaled_cov)
        print("Cov")
        for s in scaled_cov:
            print(*[ "{: >9.5f}".format(_s) for _s in s])
        print("Evalues")
        print(*[ "{: >9.5f}".format(_s) for _s in v])
        print("Evecs")
        for s in w:
            print(*[ "{: >9.5f}".format(_s) for _s in s])
        # print(scaled_cov)
        # print(v)
        # print(w.T)
        raise e


    # extract the mean lrQ part
    lrQ = z[:SS]
    dquat = z[-4:]
    for i in range(L):
        Z[1 + i][:SS] = t.compose_lrQ(lrQ, t.lrrv2lrQ(dz[i,:SM]))
        Z[1 + i][SS:SS+2] = Z[0,SS:SS+2] + dz[i,SS:SS+2]
        Z[1 + i][-4:] = t.compose_quat(dquat, t.rotvec2quat(dz[i,-3:]))

        Z[1 + i + L][:SS] = t.compose_lrQ(lrQ, t.lrrv2lrQ(-dz[i,:SM]))
        Z[1 + i + L][SS:SS+2] = Z[0,SS:SS+2] - dz[i,SS:SS+2]
        Z[1 + i + L][-4:] = t.compose_quat(dquat, t.rotvec2quat(-dz[i,-3:]))

    # Transform to new vectors
    def T(_z):
        # initial quaternion part
        quat = _z[3:SS]
        dx = dead_reckon_step( quat=quat, dl=z[SS+0], dr=z[SS+1], vave=step[7], vdiff=step[8] )
        # Post rotation quaternion part
        nquat = t.compose_quat(quat,_z[-4:])

        # Note, this is basically a compose_lrQ(...) call. dx = quat^-1 @ [dx_body, dy_body, 0]
        #  so z_new = compose_lrQ(z, [ dx_body, dy_body, 0, dquat ]) = [ z_t + quat^-1@[dx_body, dy_body, 0], quat * dquat ]
        return np.hstack([_z[:3] + dx, nquat]);

    Y = np.array([ T(_z) for _z in Z ])

    LAMBDA = L_PLUS_LAMBDA - L
    W0 = LAMBDA / L_PLUS_LAMBDA
    W = 1 / ( 2 * L_PLUS_LAMBDA )

    ## Iteratively calculate the quaternion mean
    y_bar = Y[0]
    EPS = 1e-10
    MAX = 100
    i = 0
    error = 1
    while ( i < MAX and error > EPS ):
        i += 1
        y_bar_inv = t.invert_lrQ(y_bar)
        dY = np.array([t.lrQ2lrrv(t.compose_lrQ(y_bar_inv, y)) for y in Y])
        e = (dY[0]*W0 + np.sum(dY[1:]*W,axis=0))
        error = np.linalg.norm(e) # TODO : should this be normalized?
        y_bar = t.compose_lrQ(y_bar, t.lrrv2lrQ(e))


    cov = W0 * np.array([dY[0]]).T @ np.array([dY[0]]) # exterior product
    for k in range(1,len(dY)):
        cov = cov + W * np.array([dY[k]]).T @ np.array([dY[k]]) # exterior product



    x_tilde = T(z)                                       # transform the mean
    # x_tilde = np.hstack([y_pos_bar, q_bar])              # mean of transform
    P_tilde = cov

    return x_tilde, P_tilde

def dead_reckon(x_hat, P_hat, steps):
    for step in steps:
        x_hat, P_hat = dead_reckon_apply(x_hat, P_hat, step)
    return x_hat, P_hat

class DeadReckonQueue():
    def __init__( self, max_size=100):
        self.max_size = max_size
        self.reckon_steps = []

        self.reset_reference()

    def accumulate_to(self, timestamp):
        if (len(self.reckon_steps) < 1):
            return False, self.x_hat, self.P

        steps_before_ts = []
        steps_after_ts = []

        # split the array into "before" and "after"
        idx = 0
        for step in self.reckon_steps:
            if timestamp < step[0]:
                break
            idx += 1

        before = self.reckon_steps[:idx]
        self.reckon_steps = self.reckon_steps[idx:]

        self.x_hat, self.P = dead_reckon(self.x_hat, self.P, before)

        return True, self.x_hat, self.P

    def transform_between(self, start_timestamp, end_timestamp, start_x_hat=np.array([0,0,0, 1,0,0,0]), start_P=np.eye(6)*1e-8):
        if (len(self.reckon_steps) < 1):
            return False, None, None

        steps_between_ts = []

        for step in self.reckon_steps:
            if (step[0] <= end_timestamp and step[0] >= start_timestamp):
                steps_between_ts.append(step)

        x_hat = start_x_hat.copy()
        P = start_P.copy()

        x_hat, P = dead_reckon(x_hat, P, steps_between_ts)

        return True, x_hat, P


    def reset_reference(self):
        # Full state: x,y,z, a,b,c,d
        self.x_hat = np.array([0,0,0, 1,0,0,0])
        # Manifold deviation: x,y,z (in terminal coordinates) b,c,d (relative to terminal coordinates)
        self.P = np.eye(6)*1e-8

    def push_step(self, timestamp=0, dl=0, dr=0, Dq=[1,0,0,0], vave=0, vdiff=0):
        self.reckon_steps.append(DeadReckonStep(timestamp, dl, dr, Dq, vave, vdiff))
        self.reckon_steps = self.reckon_steps[-self.max_size:]
