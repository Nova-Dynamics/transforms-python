"""Transformations

All transformations are of cooridanates NOT vectors

'sr' means shift then rotate, so that the "shift" (what I add to a vector to put it into the new coordinate reference frame) is in the initial coordinate reference frame
'lr' means location then rotate, so that the "location" (where the terminal coordinate's origin is relative to the initial) is in the initial coordinate reference frame

Types:
    euler   - [ yaw, pitch, roll ]
    quat    - [ re, i, j, k ]
    redquat - [ i, j, k ] (where re is assumed positive)
    rotmat  - 3x3
    homo    - 4x4
    srq     - [ x, y, z, i, j, k]
    sre     - [ x, y, z, yaw, pitch, roll ]
    lrq     - [ x, y, z, i, j, k]
    lre     - [ x, y, z, yaw, pitch, roll ]
    lrQ     - [ x, y, z, re, i, j, k]
"""
import sympy as sp

# ---------------------
#  Rotation conversion
# ---------------------

def quat2redquat(q):
    # sign = -1 if q[0] < 0 else 1
    # return [sp.sign(q[0])*v for v in q[1:]]
    # FIXME : does this fail?
    return q[1:]

def redquat2quat(rq):
    return [
        sp.sqrt(1 - rq[0]*rq[0] - rq[1]*rq[1] - rq[2]*rq[2]),
        rq[0],
        rq[1],
        rq[2]
    ]

def euler2quat(euler):
    sy, cy = sp.sin(euler[0]*0.5), sp.cos(euler[0]*0.5)
    Sp, cp = sp.sin(euler[1]*0.5), sp.cos(euler[1]*0.5)
    sr, cr = sp.sin(euler[2]*0.5), sp.cos(euler[2]*0.5)

    return [
         sy*Sp*sr + cy*cp*cr,
         cy*Sp*cr + sy*cp*sr,
        -sy*Sp*cr + cy*cp*sr,
         cy*Sp*sr - sy*cp*cr
    ]

def quat2euler(quat):
    # norm = np.linalg.norm(quat)
    a, b, c, d = quat#/norm
    return [
        sp.atan2( 2*(b*c-a*d), 2*(a*a+c*c)-1 ), # yaw
        sp.asin(2*(c*d+a*b)), # pitch
        sp.atan2( 2*(a*c-b*d), 2*(a*a+d*d)-1 ) # roll
    ]

# ------------------------------------------
#  Applications
# ------------------------------------------

def apply_redquat(rq,v):
    return apply_quat(redquat2quat(rq),v)

def apply_quat(q,v):
    a =             - v[0]*q[1] - v[1]*q[2] - v[2]*q[3]
    b =             + v[0]*q[0] + v[1]*q[3] - v[2]*q[2]
    c =             + v[1]*q[0] + v[2]*q[1] - v[0]*q[3]
    d =             + v[2]*q[0] + v[0]*q[2] - v[1]*q[1]
    return [
        q[0]*b - q[1]*a - q[2]*d + q[3]*c,
        q[0]*c - q[2]*a - q[3]*b + q[1]*d,
        q[0]*d - q[3]*a - q[1]*c + q[2]*b
    ]

def apply_lrq(lrq,v):
    return  apply_redquat(lrq[3:], [ a-b for a,b in zip(v,lrq[:3]) ] )

def apply_lrQ(lrQ,v):
    return  apply_quat(lrQ[3:], [ a-b for a,b in zip(v,lrQ[:3]) ] )



# -------------
#  Inversions
# -------------

def invert_redquat(rq):
    return [ -v for v in rq ]

def invert_quat(q):
    return [
        q[0],
        -q[1],
        -q[2],
        -q[3]
    ]

def invert_lrq(lrq):
    rqinv = invert_redquat(lrq[3:])
    location = [ -v for v in apply_redquat(lrq[3:], lrq[:3]) ]
    return [
        location[0],
        location[1],
        location[2],
        rqinv[0],
        rqinv[1],
        rqinv[2],
    ]

def invert_lrQ(lrQ):
    rinv = invert_quat(lrQ[3:])
    location = [ -v for v in apply_quat(lrQ[3:], lrQ[:3]) ]
    return [
        location[0],
        location[1],
        location[2],
        rinv[0],
        rinv[1],
        rinv[2],
        rinv[3],
    ]

def invert_euler(euler):
    return quat2euler(invert_quat(euler2quat(euler)))

def invert_lre(lre):
    return lrq2lre(invert_lrq(lre2lrq(lre)))
# ------------------------------------------
#  Compositions
# ------------------------------------------

def compose_redquat(rq1,rq2):
    return quat2redquat(compose_quat(redquat2quat(rq1),redquat2quat(rq2)))

def compose_quat(q1,q2):
    return [
        q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3],
        q1[0]*q2[1] + q1[1]*q2[0] + q1[2]*q2[3] - q1[3]*q2[2],
        q1[0]*q2[2] + q1[2]*q2[0] + q1[3]*q2[1] - q1[1]*q2[3],
        q1[0]*q2[3] + q1[3]*q2[0] + q1[1]*q2[2] - q1[2]*q2[1]
    ]

def compose_lrq(lrq1, lrq2):
    rq1 = lrq1[3:]
    rq3 = compose_redquat(rq1, lrq2[3:])
    location3 = [ a+b for a,b in zip(lrq1[:3],apply_redquat(invert_redquat(rq1), lrq2[:3])) ]
    return [
        location3[0],
        location3[1],
        location3[2],
        rq3[0],
        rq3[1],
        rq3[2],
    ]

def compose_lrQ(lrQ1, lrQ2):
    q1 = lrQ1[3:]
    q3 = compose_quat(q1, lrQ2[3:])
    location3 = [ a+b for a,b in zip(lrQ1[:3],apply_quat(invert_quat(q1), lrQ2[:3])) ]
    return [
        location3[0],
        location3[1],
        location3[2],
        q3[0],
        q3[1],
        q3[2],
        q3[3],
    ]

def compose_lre(lre1, lre2):
    return lrq2lre(compose_lrq(lre2lrq(lre1), lre2lrq(lre2)))


def lre2lrq(lre):
    q = quat2redquat(euler2quat(lre[3:]))
    return [
        lre[0],
        lre[1],
        lre[2],
        q[0],
        q[1],
        q[2],
    ]

def lrq2lre(lrq):
    e = quat2euler(redquat2quat(lrq[3:]))
    return [
        lrq[0],
        lrq[1],
        lrq[2],
        e[0],
        e[1],
        e[2],
    ]

def lre2lrQ(lre):
    q = euler2quat(lre[3:])
    return [
        lre[0],
        lre[1],
        lre[2],
        q[0],
        q[1],
        q[2],
        q[3],
    ]

def lrQ2lre(lrQ):
    e = quat2euler(lrQ[3:])
    return [
        lrQ[0],
        lrQ[1],
        lrQ[2],
        e[0],
        e[1],
        e[2],
    ]
