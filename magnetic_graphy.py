import numpy as np
from mayavi import mlab

def dipole(m, r, r0):
    """Calculate a field in point r created by a dipole moment m located in r0.
    Spatial components are the outermost axis of r and returned B.
    """
    # we use np.subtract to allow r and r0 to be a python lists, not only np.array
    R = np.subtract(np.transpose(r), r0).T

    # assume that the spatial components of r are the outermost axis
    norm_R = np.sqrt(np.einsum("i...,i...", R, R))

    # calculate the dot product only for the outermost axis,
    # that is the spatial components
    m_dot_R = np.tensordot(m, R, axes=1)

    # tensordot with axes=0 does a general outer product - we want no sum
    B = 3 * m_dot_R * R / norm_R**5 - np.tensordot(m, 1 / norm_R**3, axes=0)

    # include the physical constant
    B *= 1e-7

    return B

X = np.linspace(-10, 10, 200)
Y = np.linspace(-10, 10, 200)
Z = np.linspace(-10, 10, 200)
r = np.meshgrid(X, Y, Z)
r0 = [0,0,0]
m = [0.0, 0.0, 0.01]
Bx, By, Bz = dipole(m=m, r=r, r0=r0)
mlab.quiver3d(r[0],r[1],r[2],Bx,By,Bz,color=(0,0,1))
mlab.plot3d([r0[0],m[0]],[r0[1],m[1]],[r0[2],m[2]],tube_radius=0.001,color=(1,1,0))
#mlab.plot3d([r0[0],-m[0]],[r0[1],-m[1]],[r0[2],-m[2]],line_width=0.1)
