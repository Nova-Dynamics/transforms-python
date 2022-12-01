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
    lrq     - [ x, y, z, i, j, k ]
    lre     - [ x, y, z, yaw, pitch, roll ]
    lrQ     - [ x, y, z, re, i, j, k ]
"""
import numpy as np

# Pull c-wrappers for faster usage
import euclidean_wrapper as wrap

def convert_quat_to_redquat(a):
    out = np.zeros(3)
    wrap.convert_quat_to_redquat(a,out)
    return out
quat2redquat = convert_quat_to_redquat

def convert_redquat_to_quat(a):
    out = np.zeros(4)
    wrap.convert_redquat_to_quat(a,out)
    return out
redquat2quat = convert_redquat_to_quat

def convert_lrq_to_lrQ(a):
    out = np.zeros(7)
    wrap.convert_lrq_to_lrQ(a,out)
    return out
lrq2lrQ = convert_lrq_to_lrQ

def convert_lrQ_to_lrq(a):
    out = np.zeros(6)
    wrap.convert_lrQ_to_lrq(a,out)
    return out
lrQ2lrq = convert_lrQ_to_lrq

def apply_quat(a,b):
    out = np.zeros(3)
    wrap.apply_quat(a,b,out)
    return out

def apply_redquat(a,b):
    out = np.zeros(3)
    wrap.apply_redquat(a,b,out)
    return out

def apply_lrQ(a,b):
    out = np.zeros(3)
    wrap.apply_lrQ(a,b,out)
    return out

def apply_lrq(a,b):
    out = np.zeros(3)
    wrap.apply_lrq(a,b,out)
    return out

def invert_quat(a):
    out = np.zeros(4)
    wrap.invert_quat(a,out)
    return out

def invert_redquat(a):
    out = np.zeros(3)
    wrap.invert_redquat(a,out)
    return out

def invert_lrQ(a):
    out = np.zeros(7)
    wrap.invert_lrQ(a,out)
    return out

def invert_lrq(a):
    out = np.zeros(6)
    wrap.invert_lrq(a,out)
    return out

def compose_quat(a,b):
    out = np.zeros(4)
    wrap.compose_quat(a,b,out)
    return out

def compose_redquat(a,b):
    out = np.zeros(3)
    wrap.compose_redquat(a,b,out)
    return out

def compose_lrQ(a,b):
    out = np.zeros(7)
    wrap.compose_lrQ(a,b,out)
    return out

def compose_lrq(a,b):
    out = np.zeros(6)
    wrap.compose_lrq(a,b,out)
    return out

# Keep all the "slow" functions, until we get around to wrapping them too

# ---------------------
#  Rotation conversion
# ---------------------

def euler2rotmat(euler):
    sy, cy = np.sin(euler[0]), np.cos(euler[0])
    sp, cp = np.sin(euler[1]), np.cos(euler[1])
    sr, cr = np.sin(euler[2]), np.cos(euler[2])

    return np.array([
        [ cr*cy + sr*sp*sy, -cr*sy + sr*sp*cy, -sr*cp ],
        [            cp*sy,             cp*cy,     sp ],
        [ sr*cy - cr*sp*sy, -sr*sy - cr*sp*cy,  cr*cp ]
    ])

def rotmat2euler(R):
    return np.array([
        np.arctan2( R[1,0], R[1,1] ), # yaw
        np.arcsin(np.clip(R[1,2],-1,1)), # pitch
        np.arctan2( -R[0,2], R[2,2] ) # roll
    ])


def euler2quat(euler):
    sy, cy = np.sin(euler[0]*0.5), np.cos(euler[0]*0.5)
    sp, cp = np.sin(euler[1]*0.5), np.cos(euler[1]*0.5)
    sr, cr = np.sin(euler[2]*0.5), np.cos(euler[2]*0.5)

    return np.array([
         sy*sp*sr + cy*cp*cr,
         cy*sp*cr + sy*cp*sr,
        -sy*sp*cr + cy*cp*sr,
         cy*sp*sr - sy*cp*cr
    ])

def quat2redquat(q):
    sign = -1 if q[0] < 0 else 1
    return sign*q[1:]

def redquat2quat(rq):
    return np.array([
        np.sqrt(np.clip(1 - rq[0]*rq[0] - rq[1]*rq[1] - rq[2]*rq[2],0,1)),
        rq[0],
        rq[1],
        rq[2]
    ])

def redquat2euler(redquat):
    return quat2euler(redquat2quat(redquat))

def quat2euler(quat):
    # norm = np.linalg.norm(quat)
    a, b, c, d = quat#/norm
    return np.array([
        # np.arctan2( 2*(-ad+bc), -1+2*(a*a+c*c) ), # yaw
        # np.arcsin(np.clip(2*(a*b+c*d), -1, 1)), # pitch
        # np.arctan2( 2*(-ad+bc), -1+2*(a*a+c*c) ) # roll
        np.arctan2( 2*(b*c-a*d), 2*(a*a+c*c)-1 ), # yaw
        np.arcsin(np.clip(2*(c*d+a*b), -1, 1)), # pitch
        np.arctan2( 2*(a*c-b*d), 2*(a*a+d*d)-1 ) # roll
    ])

def redquat2rotmat(redquat):
    return quat2rotmat(redquat2quat(redquat))

def quat2rotmat(quat):
    a, b, c, d = quat

    asq, bsq, csq, dsq = a*a, b*b, c*c, d*d
    return np.array([
        [ asq+bsq-csq-dsq,     2*(b*c+a*d),     2*(b*d-a*c) ],
        [     2*(b*c-a*d), asq-bsq+csq-dsq,     2*(c*d+a*b) ],
        [     2*(b*d+a*c),     2*(c*d-a*b), asq-bsq-csq+dsq ]
    ])

def rotmat2quat(R):
    asq = (1 + R[0,0] + R[1,1] + R[2,2]) * 0.25
    b = np.sqrt(np.clip((R[0,0] + 1) * 0.5 - asq,0,1))
    c = np.sqrt(np.clip((R[1,1] + 1) * 0.5 - asq,0,1))
    d = np.sqrt(np.clip((R[2,2] + 1) * 0.5 - asq,0,1))

    if asq > 1e-12:
        b *= -1 if (R[1,2] - R[2,1] < 0) else 1
        c *= -1 if (R[2,0] - R[0,2] < 0) else 1
        d *= -1 if (R[0,1] - R[1,0] < 0) else 1
    elif b > 1e-12:
        c *= -1 if (R[0,1] + R[1,0] < 0) else 1
        d *= -1 if (R[2,0] + R[0,2] < 0) else 1
    elif c > 1e-12:
        d *= -1 if (R[1,2] + R[2,1] < 0) else 1
    
    return np.array([np.sqrt(np.clip(asq,0,1)),b,c,d])

# ------------------------------------------
#  Applications
# ------------------------------------------

def apply_rotmat(R,v):
    return R @ v

def apply_redquat(rq,v):
    return apply_quat(redquat2quat(rq),v)

def apply_quat(q,v):
    a =             - v[0]*q[1] - v[1]*q[2] - v[2]*q[3]
    b =             + v[0]*q[0] + v[1]*q[3] - v[2]*q[2]
    c =             + v[1]*q[0] + v[2]*q[1] - v[0]*q[3]
    d =             + v[2]*q[0] + v[0]*q[2] - v[1]*q[1]
    return np.array([
        q[0]*b - q[1]*a - q[2]*d + q[3]*c,
        q[0]*c - q[2]*a - q[3]*b + q[1]*d,
        q[0]*d - q[3]*a - q[1]*c + q[2]*b
    ])

def apply_euler(euler,v):
    return apply_quat(euler2quat(euler), v)

def apply_homo(H,v):
    _v = np.array([v[0],v[1],v[2],1])
    return (H @ _v)[:3]

def apply_srq(srq,v):
    return  apply_redquat(srq[3:], v + srq[:3])

def apply_lrq(lrq,v):
    return apply_redquat(lrq[3:], v - lrq[:3])

def apply_sre(sre,v):
    return apply_euler(sre[3:], v + sre[:3])

def apply_lre(lre,v):
    return apply_euler(lre[3:], v - lre[:3])

def apply_lrQ(lrQ,v):
    return apply_quat(lrQ[3:], v - lrQ[:3])


# -------------
#  Inversions
# -------------

def invert_rotmat(R):
    return np.linalg.inv(R)

def invert_euler(euler):
    return quat2euler(invert_quat(euler2quat(euler)))

def invert_homo(H):
    return np.linalg.inv(H)

def invert_srq(srq):
    rqinv = invert_redquat(srq[3:])
    shift = -apply_redquat(srq[3:], srq[:3])
    return np.array([
        shift[0],
        shift[1],
        shift[2],
        rqinv[0],
        rqinv[1],
        rqinv[2],
    ])

def invert_sre(sre):
    return homo2sre(invert_homo(sre2homo(sre)))

def invert_lre(lre):
    return homo2lre(invert_homo(lre2homo(lre)))

# ------------
#  Euclidian
# ------------

def srq2homo(srq):
    shift = srq[:3]
    R = redquat2rotmat(srq[3:])

    H = np.eye(4)
    H[:3,:3] = R
    H[:3,3] = apply_rotmat(R, shift) # map shift to shift_prime

    return H

def lrq2homo(lrq):
    shift = -np.array(lrq[:3])
    R = redquat2rotmat(lrq[3:])

    H = np.eye(4)
    H[:3,:3] = R
    H[:3,3] = apply_rotmat(R, shift) # map shift to shift_prime

    return H

def lrQ2homo(lrQ):
    shift = -np.array(lrQ[:3])
    R = quat2rotmat(lrQ[3:])

    H = np.eye(4)
    H[:3,:3] = R
    H[:3,3] = apply_rotmat(R, shift) # map shift to shift_prime

    return H

def sre2homo(sre):
    shift = sre[:3]
    R = euler2rotmat(sre[3:])

    H = np.eye(4)
    H[:3,:3] = R
    H[:3,3] = apply_rotmat(R, shift) # map shift to shift_prime

    return H

def lre2homo(lre):
    shift = -np.array(lre[:3])
    R = euler2rotmat(lre[3:])

    H = np.eye(4)
    H[:3,:3] = R
    H[:3,3] = apply_rotmat(R, shift) # map shift to shift_prime

    return H

def homo2srq(H):
    rq = rotmat2quat(H[:3,:3])[1:]
    shift = apply_redquat(-rq, H[:3,3]) # map shift_prime to shift

    return np.array([
        shift[0],
        shift[1],
        shift[2],
        rq[0],
        rq[1],
        rq[2]
    ])

def homo2lrq(H):
    rq = rotmat2quat(H[:3,:3])[1:]
    shift = apply_redquat(-rq, H[:3,3]) # map shift_prime to shift

    return np.array([
        -shift[0],
        -shift[1],
        -shift[2],
        rq[0],
        rq[1],
        rq[2]
    ])

def homo2lrQ(H):
    r = rotmat2quat(H[:3,:3])[1:]
    shift = apply_quat(-r, H[:3,3]) # map shift_prime to shift

    return np.array([
        -shift[0],
        -shift[1],
        -shift[2],
        r[0],
        r[1],
        r[2],
        r[3]
    ])


def homo2sre(H):
    e = rotmat2euler(H[:3,:3])
    shift = apply_rotmat(np.linalg.inv(H[:3,:3]), H[:3,3]) # map shift_prime to shift

    return np.array([
        shift[0],
        shift[1],
        shift[2],
        e[0],
        e[1],
        e[2]
    ])

def homo2lre(H):
    e = rotmat2euler(H[:3,:3])
    shift = apply_rotmat(np.linalg.inv(H[:3,:3]), H[:3,3]) # map shift_prime to shift

    return np.array([
        -shift[0],
        -shift[1],
        -shift[2],
        e[0],
        e[1],
        e[2]
    ])

def lre2lrQ(lre):
    q = euler2quat(lre[3:])

    return np.array([
        lre[0],
        lre[1],
        lre[2],
        q[0],
        q[1],
        q[2],
        q[3]
    ])

def lrQ2lre(lrQ):
    e = quat2euler(lrQ[3:])

    return np.array([
        lrQ[0],
        lrQ[1],
        lrQ[2],
        e[0],
        e[1],
        e[2],
    ])

# ------------------------------------------
#  Compositions
# ------------------------------------------

def compose_rotmat(R1,R2):
    return R2 @ R1

def compose_homo(H1,H2):
    return H2 @ H1

def compose_srq(srq1, srq2):
    rq1 = srq1[3:]
    rq3 = compose_redquat(rq1, srq2[3:])
    shift3 = srq1[:3] + apply_redquat(invert_redquat(rq1), srq2[:3])
    return np.array([
        shift3[0],
        shift3[1],
        shift3[2],
        rq3[0],
        rq3[1],
        rq3[2],
    ])

def compose_sre(sre1, sre2):
    return homo2sre(compose_homo(sre2homo(sre1), sre2homo(sre2)))

def compose_lre(lre1, lre2):
    return homo2lre(compose_homo(lre2homo(lre1), lre2homo(lre2)))
