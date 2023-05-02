
import numpy as np
import matplotlib.pyplot as plt
import robotransforms.euclidean.euclidean_pure as t
from scipy.spatial.transform import Rotation as R

def plot_quaternions(quaternions, nrows=1):
    n_quaternions = quaternions.shape[0]
    ncols = int(np.ceil(n_quaternions / nrows))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3*ncols, 3*nrows))
    axes = axes.flatten()

    for i in range(n_quaternions):
        quat = quaternions[i]
        r = R.from_quat(quat)
        yaw = r.as_euler('zyx')[2]

        axes[i].arrow(0, 0, np.cos(yaw), np.sin(yaw), head_width=0.1, head_length=0.1, fc='k', ec='k')
        axes[i].set_xlim(-1.2, 1.2)
        axes[i].set_ylim(-1.2, 1.2)
        axes[i].set_aspect('equal')
        axes[i].set_title(f'Quaternion {i+1}')

    for i in range(n_quaternions, nrows*ncols):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

## ======================
## CODE
## ======================
def DeadReckonStep(timestamp, dl, dr, Dq): #, vave, vdiff):
    return np.array([ timestamp, dl, dr, Dq[0], Dq[1], Dq[2], Dq[3]]) #, vave, vdiff ])

L_PLUS_LAMBDA = 3

def sinc(x):
    # Use power series for small x
    if (np.abs(x) < 1e-4): return 1 - (x*x/6) + (x*x*x*x/120)
    return np.sin(x)/x

D = 0.5
def dead_reckon_step( quat, dl, dr): #, vave, vdiff ):
    lstar = 0.5 * ( dl + dr )
    to2 = ( dl - dr ) / ( 2 * D )
    chord = lstar * sinc(to2)
    sin = np.sin(to2)
    cos = np.cos(to2)

    dx = chord * sin
    dy = chord * cos

    return t.apply_quat(t.invert_quat(quat), [dx,dy,0])

def dead_reckon_step_errors( dl, dr, dl_scale=0.005): #vave, vdiff, dl_scale=0.005):
    ddl = (dl_scale * np.abs(dl))
    ddr = (dl_scale * np.abs(dr))
    rho=1-1e-3
    #rho = 1-1e-8
    #if ( np.abs(vdiff) >= 1e-3 ) :
    #    rho = np.abs(np.tanh( vave / vdiff ))

    #  [ var_dl, cov_dldr, var_dr, var_b, var_c, var_d ]
    return [ ddl*ddl + 1e-8, rho*ddl*ddr, ddr*ddr + 1e-8 ] # TODO : these are estimates, based on gps's 1Hz

# rotvec is RV from
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6339217/pdf/sensors-19-00149.pdfh
def quat2rotvec(quat):
    rotvec = np.zeros(3)
    qvec_norm = np.linalg.norm(quat[1:])
    # TODO: Figure out what to do otherwise
    if quat[0] >= 0:
        rotvec = 2 * np.arcsin(qvec_norm) * quat[1:]
    else:
        # Figure out what to do here...
        rotvec = None

    return rotvec

def rotvec2quat(rotvec):
    rotvec_norm = np.linalg.norm(rotvec)
    if rotvec_norm <= np.pi:
        quat = np.zeros(4)
        quat[0] = np.cos(rotvec_norm / 2)
        quat[1:] = np.sin(rotvec_norm / 2) * rotvec
    else:
        # Figure out what to do here...
        quat = None

    return quat

def dead_reckon_apply(x_hat, P_hat, step, j):

    # No-op if we didn't move
    if (step[1] == 0 and step[2] == 0):
        return x_hat, P_hat

    # Build up extended state vector
    SS = 4
    z = np.zeros((SS+2))
    z[:SS] = x_hat[:]
    z[SS+0] = step[1]
    z[SS+1] = step[2]

    # Build up extended covariance on manifold
    SM = 3
    P_z = np.zeros((SM+2,SM+2))
    P_z[:SM,:SM] = P_hat[:SM,:SM]
    #                            dl,      dr,      vave,    vdiff
    e = dead_reckon_step_errors( step[1], step[2]) #, step[7], step[8] )
    P_z[0+SM][0+SM] = e[0] # var_dl
    P_z[0+SM][1+SM] = e[1] # cov_dldr
    P_z[1+SM][0+SM] = e[1] # cov_dldr
    P_z[1+SM][1+SM] = e[2] # var_dr

    # Generate sigma points along manifold from enclosing deviation vectors with the state
    L = SM+2
    Z = np.zeros((2*L+1,SS+2))
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

    #                x_hat y_hat z  a     b  c  d
    lrQ = np.array([z[0], z[1],  0, z[2], 0 ,0, z[3]])
    print("lrQ", lrQ)
    # Computing sigma points
    for i in range(L):
        # The first SS are gotten via composition
        dlrQ = np.array([dz[i][0], dz[i][1], 0, np.sqrt(1-dz[i][2]**2), 0 ,0, dz[i][2]])
        lrQ1 = t.compose_lrQ(lrQ, dlrQ)
        #                         x        y        a        d
        Z[1 + i][:SS] = np.array([lrQ1[0], lrQ1[1], lrQ1[3], lrQ1[6]])
        dlrQ = np.array([-dz[i][0], -dz[i][1], 0, np.sqrt(1-dz[i][2]**2), 0, 0, -dz[i][2]])
        lrQ2 = t.compose_lrQ(lrQ, dlrQ)
        Z[1 + i + L][:SS] = np.array([lrQ2[0], lrQ2[1], lrQ2[3], lrQ2[6]])

        # The last 2 (dl, dr) are gotten via addition
        Z[1 + i][SS:] = z[SS:] + dz[i][SM:]
        Z[1 + i + L][SS:] = z[SS:] - dz[i][SM:]
    print("Z")
    for row in Z:
        print(",".join("{:12.8f}".format(v) for v in row)) #, np.sqrt(row[2]**2+row[3]**2))
    print()

    # Transform to new vectors
    def T(_z):
        #               a       b  c  d
        quat = np.array([_z[2], 0 ,0, _z[3]])
        dx = dead_reckon_step( quat=quat, dl=z[SS+0], dr=z[SS+1]) #, vave=step[7], vdiff=step[8] )
        # Taking current quat and applying step quat rotation to it
        next_quat = t.compose_quat(quat, step[3:7])
        # Only return offset, this is marginalizing out all the quat dependances
        # return _z[:3] + dx
        return np.hstack([_z[:2] + dx[:2], next_quat[0], next_quat[3]])

    Y = np.array([ T(_z) for _z in Z ])
    print("Y")
    for row in Y:
        print(",".join("{:12.8f}".format(v) for v in row))#, np.sqrt(row[2]**2+row[3]**2))
    print()

    M = len(Y[0])
    LAMBDA = L_PLUS_LAMBDA - L
    W0 = LAMBDA / L_PLUS_LAMBDA
    W = 1 / ( 2 * L_PLUS_LAMBDA )

    # Grab x and y in every sigma point
    Y_pos = Y[:,:2]

    # Take a weighted average of the position components
    y_pos_bar = (Y_pos[0]*W0 + np.sum(Y_pos[1:]*W,axis=0))[:2]
    
    #Y_quat = np.array([[a,0,0,d] for a,d in zip(Y[:,2],Y[:,3])])
    #print("Y_quat shape:", Y_quat.shape)
    #Y_euler = np.array([t.quat2euler(q) for q in Y_quat])
    #print("Y_euler shape:", Y_euler.shape)
    #y_euler_bar = (Y_euler[0]*W0 + np.sum(Y_euler[1:]*W,axis=0))
    #print("Y_euler_bar shape:", y_euler_bar.shape)
    #y_euler_diff = Y_euler-y_euler_bar
    #print("Y_euler_diff shape:", y_euler_diff.shape)

    #dy_quat_bar = np.array([t.quat2redquat(t.euler2quat(q)) for q in y_euler_diff])
    #print("dy_quat_bar shape:", dy_quat_bar.shape)
    #print("Y_pos - y_pos_bar shape:", (Y_pos - y_pos_bar).shape)
    #print("dy_quat_bar[:,:1] shape:", dy_quat_bar[:,:1].shape)
    #print("dy_quat_bar[:,2:]", dy_quat_bar[:,2:].shape)

    #dY = np.hstack([Y_pos - y_pos_bar, dy_quat_bar[:,2:]])
    #print("dY shape:", dY.shape)

    ## Iteratively calculate the quaternion mean
    Y_quat = np.array([[a,0,0,d] for a,d in zip(Y[:,2],Y[:,3])])
    #print("Y_quat", Y_quat)
    q_bar = Y_quat[0]
    q_bar_inv = t.invert_quat(Y_quat[0])
    EPS = 1e-12
    MAX = 100
    i = 0
    error = 1
    while ( i < MAX and error > EPS ):
        i += 1
        e_rquats = np.array([t.quat2redquat(t.compose_quat(q_bar_inv, y)) for y in Y_quat])
        e = (e_rquats[0]*W0 + np.sum(e_rquats[1:]*W,axis=0))
        error = np.linalg.norm(e)
        q_bar = t.compose_quat(q_bar, t.redquat2quat(e))
        q_bar_inv = t.invert_quat(q_bar)

    print(i, error)

    dY = np.hstack([Y_pos-y_pos_bar, e_rquats[:,2:]]) 

    #dY = Y - y_bar
    
    

    # for row in dY:
    #     print(",".join("{:12.8f}".format(v) for v in row))
    # print()
    print("dY")
    for row in dY:
        print(",".join("{:12.8f}".format(v) for v in row))
    print()
    cov = W0 * np.array([dY[0]]).T @ np.array([dY[0]]) # exterior product
    for k in range(1,len(dY)):
        cov = cov + W * np.array([dY[k]]).T @ np.array([dY[k]]) # exterior product
 
    print("cov")
    for row in cov:
        print(",".join("{:12.8f}".format(v) for v in row))
    print()
    x_tilde = T(z)                                       # transform the mean
    P_tilde = cov

    for zz in Z:
        quat = np.array([zz[2], 0,0, -zz[3]])
        xyz = np.array([ t.apply_quat(quat, _xyz) for _xyz in [
            [0,0.05,0],
            [-0.01,0,0],
            [0.01,0,0],
            [0,0.05,0]
        ]]) + np.array([zz[0],zz[1],0])
        plt.fill(xyz[:,0],xyz[:,1],color="green", alpha=0.5)
    for zz in Y:
        quat = np.array([zz[2], 0,0, -zz[3]])
        xyz = np.array([ t.apply_quat(quat, _xyz) for _xyz in [
            [0,0.05,0],
            [-0.01,0,0],
            [0.01,0,0],
            [0,0.05,0]
        ]]) + np.array([zz[0],zz[1],0])
        plt.fill(xyz[:,0],xyz[:,1],color="blue", alpha=0.5)

    #if j > 10:
    #    plt.show()
    #    plot_quaternions(Y_quat, 3)

    return x_tilde, P_tilde

def dead_reckon(x_hat, P_hat, steps, j):
    for i, step in enumerate(steps):
        x_hat, P_hat = dead_reckon_apply(x_hat, P_hat, step, j)
    return x_hat, P_hat

## ======================
## TEST
## ======================

D = 0.464
dl = 0.08
dr = 0.07
to2 = ( dl - dr ) / ( 2 * D )

#                 x,y,a,d
x_hat = np.array([0,0,1,0])
P_hat = np.diag([1e-1,1e-1,0.05])
step = np.array([
    43994.76,          # ts
    dl, dr,      # dl, dr
    np.cos(to2), 0, 0, -np.sin(to2),    # Dq: a,b,c,d
    #0.215, 0.143,     # vave, vdiff
])
steps = 3*[step]


xhs = [x_hat]
phs = [P_hat]

for i in range(50):
    print("==================================== "+str(i+1))
    x_hat, P_hat = dead_reckon(x_hat, P_hat, steps, i)
    xhs.append(x_hat)
    phs.append(P_hat)

    v,w = np.linalg.eigh(P_hat)
    print("v")
    print(",".join("{:12.8f}".format(v) for v in v))
    print()
    print("W")
    for row in w:
        print(",".join("{:12.8f}".format(v) for v in row))
    print()

print("x")
print(",".join("{:12.8f}".format(v) for v in x_hat))
print()
print("P")
for row in P_hat:
    print(",".join("{:12.8f}".format(v) for v in row))
print()


fig = plt.figure()
ax = fig.add_subplot(111)

for x,p in zip(xhs, phs):
    L = np.linalg.cholesky(p)[:2,:2]
    xy = np.array([L@np.array([np.sin(u), np.cos(u)]) + x[:2] for u in np.linspace(0,2*np.pi,100)])
    plt.plot(x[:1], x[1:2], "x")
    plt.plot(xy[:,0], xy[:,1])


plt.show()

