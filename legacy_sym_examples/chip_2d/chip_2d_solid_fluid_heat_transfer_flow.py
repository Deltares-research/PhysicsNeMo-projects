"""
Explicit PyTorch PINN for 2D chip flow.
Replaces legacy domain abstractions with direct geometry sampling and explicit residuals.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse

# Geometry and config
CHANNEL_LENGTH = (-2.5, 5.0)
CHANNEL_WIDTH = (-0.5, 0.5)
CHIP_POS = -1.0
CHIP_HEIGHT = 0.6
CHIP_WIDTH = 1.0
INLET_VELOCITY = 5.24386

# Training hyperparameters (from config)
BATCH_SIZES = {
    'inlet': 64,
    'outlet': 64,
    'no_slip': 2800,
    'interior_lr': 1000,
    'interior_hr': 1000,
    'integral_continuity': 512,
    'num_integral_continuity': 4,
}
MAX_STEPS = 40000
LEARNING_RATE = 1e-3


def _sample_masked_points(batch_size, x_low, x_high, y_low, y_high, mask_fn):
    points = []
    total = 0
    while total < batch_size:
        n_try = max(batch_size * 2, 1024)
        x = np.random.uniform(x_low, x_high, (n_try, 1))
        y = np.random.uniform(y_low, y_high, (n_try, 1))
        mask = mask_fn(x, y).reshape(-1)
        if np.any(mask):
            xy = np.concatenate([x[mask], y[mask]], axis=1)
            points.append(xy)
            total += xy.shape[0]
    stacked = np.vstack(points)[:batch_size]
    return torch.tensor(stacked, dtype=torch.float32)

def sample_interior(batch_size, region='lr'):
    """Sample points inside the channel minus chip rectangle."""

    def mask_fn(x, y):
        outside_chip = ~(
            (x >= CHIP_POS)
            & (x <= CHIP_POS + CHIP_WIDTH)
            & (y >= CHANNEL_WIDTH[0])
            & (y <= CHANNEL_WIDTH[0] + CHIP_HEIGHT)
        )
        if region == 'hr':
            return outside_chip & (x > (CHIP_POS - 0.25)) & (x < (CHIP_POS + CHIP_WIDTH + 0.25))
        if region == 'lr':
            return outside_chip & ((x < (CHIP_POS - 0.25)) | (x > (CHIP_POS + CHIP_WIDTH + 0.25)))
        return outside_chip

    return _sample_masked_points(
        batch_size,
        CHANNEL_LENGTH[0],
        CHANNEL_LENGTH[1],
        CHANNEL_WIDTH[0],
        CHANNEL_WIDTH[1],
        mask_fn,
    )

def sample_inlet(batch_size):
    y = np.random.uniform(CHANNEL_WIDTH[0], CHANNEL_WIDTH[1], (batch_size, 1))
    x = np.full((batch_size, 1), CHANNEL_LENGTH[0])
    return torch.tensor(np.concatenate([x, y], axis=1), dtype=torch.float32)

def sample_outlet(batch_size):
    y = np.random.uniform(CHANNEL_WIDTH[0], CHANNEL_WIDTH[1], (batch_size, 1))
    x = np.full((batch_size, 1), CHANNEL_LENGTH[1])
    return torch.tensor(np.concatenate([x, y], axis=1), dtype=torch.float32)

def sample_no_slip(batch_size):
    # Sample on channel walls and chip boundary
    n = batch_size // 2
    # Channel walls
    x1 = np.random.uniform(CHANNEL_LENGTH[0], CHANNEL_LENGTH[1], (n, 1))
    y1 = np.full((n, 1), CHANNEL_WIDTH[0])
    x2 = np.random.uniform(CHANNEL_LENGTH[0], CHANNEL_LENGTH[1], (n, 1))
    y2 = np.full((n, 1), CHANNEL_WIDTH[1])
    # Chip boundary (vertical sides)
    x3 = np.full((n, 1), CHIP_POS)
    y3 = np.random.uniform(CHANNEL_WIDTH[0], CHANNEL_WIDTH[0] + CHIP_HEIGHT, (n, 1))
    x4 = np.full((n, 1), CHIP_POS + CHIP_WIDTH)
    y4 = np.random.uniform(CHANNEL_WIDTH[0], CHANNEL_WIDTH[0] + CHIP_HEIGHT, (n, 1))
    pts = np.concatenate([np.concatenate([x1, y1], axis=1), np.concatenate([x2, y2], axis=1),
                         np.concatenate([x3, y3], axis=1), np.concatenate([x4, y4], axis=1)], axis=0)
    return torch.tensor(pts[:batch_size], dtype=torch.float32)

class FlowNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 3)  # u, v, p
        )
    def forward(self, x):
        return self.net(x)

def navier_stokes_residual(xy, net, nu=0.004195088, rho=1.1614):
    xy.requires_grad_(True)
    out = net(xy)
    u, v, p = out[:, 0:1], out[:, 1:2], out[:, 2:3]
    grads = torch.autograd.grad(u.sum(), xy, create_graph=True)[0]
    u_x = grads[:, 0:1]
    u_y = grads[:, 1:2]
    grads = torch.autograd.grad(v.sum(), xy, create_graph=True)[0]
    v_x = grads[:, 0:1]
    v_y = grads[:, 1:2]
    grads = torch.autograd.grad(p.sum(), xy, create_graph=True)[0]
    p_x = grads[:, 0:1]
    p_y = grads[:, 1:2]
    # Continuity
    continuity = u_x + v_y
    # Momentum x
    u_xx = torch.autograd.grad(u_x.sum(), xy, create_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(u_y.sum(), xy, create_graph=True)[0][:, 1:2]
    momentum_x = u * u_x + v * u_y + p_x / rho - nu * (u_xx + u_yy)
    # Momentum y
    v_xx = torch.autograd.grad(v_x.sum(), xy, create_graph=True)[0][:, 0:1]
    v_yy = torch.autograd.grad(v_y.sum(), xy, create_graph=True)[0][:, 1:2]
    momentum_y = u * v_x + v * v_y + p_y / rho - nu * (v_xx + v_yy)
    return continuity, momentum_x, momentum_y

def train(max_steps):
    net = FlowNet()
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    for step in range(max_steps):
        # Interior (low-res)
        xy_lr = sample_interior(BATCH_SIZES['interior_lr'], region='lr')
        cont_lr, mx_lr, my_lr = navier_stokes_residual(xy_lr, net)
        loss_lr = (cont_lr**2 + mx_lr**2 + my_lr**2).mean()
        # Interior (high-res)
        xy_hr = sample_interior(BATCH_SIZES['interior_hr'], region='hr')
        cont_hr, mx_hr, my_hr = navier_stokes_residual(xy_hr, net)
        loss_hr = (cont_hr**2 + mx_hr**2 + my_hr**2).mean()
        # Inlet BC
        xy_inlet = sample_inlet(BATCH_SIZES['inlet'])
        out_inlet = net(xy_inlet)
        loss_inlet = ((out_inlet[:, 0:1] - INLET_VELOCITY)**2 + out_inlet[:, 1:2]**2).mean()
        # Outlet BC (p=0)
        xy_outlet = sample_outlet(BATCH_SIZES['outlet'])
        out_outlet = net(xy_outlet)
        loss_outlet = (out_outlet[:, 2:3]**2).mean()
        # No-slip BC
        xy_noslip = sample_no_slip(BATCH_SIZES['no_slip'])
        out_noslip = net(xy_noslip)
        loss_noslip = (out_noslip[:, 0:2]**2).mean()
        # Total loss
        loss = loss_lr + loss_hr + loss_inlet + loss_outlet + loss_noslip
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 1000 == 0:
            print(f"Step {step}: loss={loss.item():.6f}")
    torch.save(net.state_dict(), "chip2d_flow_pinn.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_steps", type=int, default=MAX_STEPS)
    args = parser.parse_args()
    train(args.max_steps)
