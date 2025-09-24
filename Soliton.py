import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting
from matplotlib import cm  # Colormap
from matplotlib.ticker import LinearLocator

# ------------------------ 2nd Derivative (Centered) ------------------------ #
def d2u_dx2(u, dx):
    d2u = np.zeros_like(u, dtype=np.complex128)
    d2u[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2]) / dx**2
    d2u[0] = (u[1] - 2*u[0] + u[0]) / dx**2  # Neumann BC (u'[0]=0)
    d2u[-1] = (u[-1] - 2*u[-1] + u[-2]) / dx**2  # Neumann BC (u'[-1]=0)
    return d2u

# ------------------------ RK4 Step for NLS ------------------------ #
def rk4_step_nls(u, dx, dt, k):
    def rhs(u):
        return 1j * (-d2u_dx2(u, dx) - k * np.abs(u)**2 * u)

    k1 = rhs(u)
    k2 = rhs(u + 0.5 * dt * k1)
    k3 = rhs(u + 0.5 * dt * k2)
    k4 = rhs(u + dt * k3)
    return u + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

# ------------------------ Initial Condition ------------------------ #
def get_initial_condition(x, L, type='bump'):
    if type == 'bump':
        return 0.01 * np.exp(-100 * (x - L/2)**2) + 0j
    elif type == 'sine':
        return 0.5 * np.sin(2 * np.pi * x / L).astype(np.complex128)
    elif type == 'soliton':
        return 1.0 / np.cosh(x - L/2)
    else:
        raise ValueError("Invalid initial condition type")

# ------------------------ Parameters ------------------------ #
k = 1.0
L = 10.0
T = 2.0
Nx = 500
Nt = 1000

x = np.linspace(0, L, Nx)
dx = x[1] - x[0]
dt = 1e-4

init_type = 'bump'
u = np.zeros((Nt, Nx), dtype=np.complex128)
u[0, :] = get_initial_condition(x, L, init_type)

# ------------------------ Time Evolution ------------------------ #
for n in range(Nt - 1):
    u[n+1, :] = rk4_step_nls(u[n, :], dx, dt, k)

# ------------------------ Animation ------------------------ #
fig, ax = plt.subplots(figsize=(10, 4))
line, = ax.plot(x, np.abs(u[0, :]), lw=2)
ax.set_xlim(x[0], x[-1])
ax.set_xlabel('x')
ax.set_ylabel('|u(x, t)|')
ax.set_title(f'NLS Evolution with {init_type} initial condition')

def update(frame):
    y = np.abs(u[frame])
    line.set_ydata(y)
    ax.set_ylim(0, 1.2 * np.max(y))  # Dynamic scaling
    ax.set_title(f"NLS Evolution (t = {frame*dt:.3f})")
    return line,

ani = FuncAnimation(fig, update, frames=Nt, interval=30, blit=True)
plt.tight_layout()
plt.show()

