
import numpy as np
import matplotlib.pyplot as plt
import robotransforms.dead_reckon as dr

D = 0.46637199585

_dl = 0.08
_dr = 0.07
to2 = ( _dl - _dr ) / ( 2 * D )

x_hat = np.array([0,0,0, 1,0,0,0])
P_hat = np.diag([1e-1,1e-1,1e-1,1e-5,1e-5,1e-2])
step = np.array([
    43994.76,          # ts
    _dl, _dr,      # dl, dr
    np.cos(to2), 0, 0, np.sin(to2),    # Dq: a,b,c,d
    1, 0,     # vave, vdiff
])
steps = 10*[step]


xhs = [x_hat]
phs = [P_hat]

for i in range(20):
    x_hat, P_hat = dr.dead_reckon_pure.dead_reckon(x_hat, P_hat, steps)
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

