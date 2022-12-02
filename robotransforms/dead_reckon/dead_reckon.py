import numpy as np

import dead_reckon_wrapper as wrap 
from . import dead_reckon_pure as pure

def DeadReckonStep(timestamp, dl, dr, Dq, vave, vdiff):
    return np.array([ timestamp, dl, dr, Dq[0], Dq[1], Dq[2], Dq[3], vave, vdiff ])

def dr_calculate_d(vave, vdiff):
    return wrap.dr_calculate_d(vave, vdiff)

def dead_reckon_step( quat, dl, dr, vave, vdiff ):
    out = np.zeros(3)
    wrap.dead_reckon_step( quat, dl, dr, vave, vdiff, out )
    return out

def dead_reckon_step_errors( dl, dr, vave, vdiff, dl_scale=0.005 ):
    out = np.zeros(3)
    wrap.dead_reckon_step_errors( dl, dr, vave, vdiff, dl_scale, out )
    return out

def dead_reckon_apply( x_hat, P_hat, step ):
    x_tilde, P_tilde = np.zeros(7), np.zeros((6,6))
    wrap.dead_reckon_apply( step, x_hat, P_hat, x_tilde, P_tilde )
    return x_tilde, P_tilde

def dead_reckon( x_hat, P_hat, steps ):
    n = len(steps)
    x_tilde, P_tilde = np.zeros(7), np.zeros((6,6))
    wrap.dead_reckon( n, steps, x_hat, P_hat, x_tilde, P_tilde )
    return x_tilde, P_tilde

class DeadReckonQueue:
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
