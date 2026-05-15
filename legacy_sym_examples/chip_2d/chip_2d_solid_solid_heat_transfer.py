"""
Explicit PyTorch PINN for solid-solid heat transfer in 2D chip geometry.
Replaces legacy abstractions with direct geometry sampling and explicit residuals.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse

# Geometry and config
CHANNEL_ORIGIN = (-2.5, -0.5)
CHANNEL_DIM = (5.0, 1.0)
HEAT_SINK_BASE_ORIGIN = (-1.0, -0.5)
HEAT_SINK_BASE_DIM = (1.0, 0.2)
FIN_ORIGIN = HEAT_SINK_BASE_ORIGIN
FIN_DIM = (1.0, 0.6)
BOX_ORIGIN = (-1.1, -0.5)
BOX_DIM = (1.2, 1.0)
SOURCE_ORIGIN = (-0.7, -0.5)
SOURCE_DIM = (0.4, 0.0)
INLET_TEMP = 25.0
SOURCE_GRAD = 0.025

BATCH_SIZES = {
    'inlet': 200,
    'outlet': 200,
    'walls': 1000,
    'interior_lr': 2000,
    'interior_hr': 2000,
    'interiorS': 1000,
    'heat_source': 100,
    'interface': 400,
    'chip_walls': 100,
}
MAX_STEPS = 100000
LEARNING_RATE = 1e-4


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

def sample_solid_I_lr(batch_size):
    # Low-res region: outside box

    def mask_fn(x, y):
        return ~(
            (x >= BOX_ORIGIN[0])
            & (x <= BOX_ORIGIN[0] + BOX_DIM[0])
            & (y >= BOX_ORIGIN[1])
            & (y <= BOX_ORIGIN[1] + BOX_DIM[1])
        )

    return _sample_masked_points(
        batch_size,
        CHANNEL_ORIGIN[0],
        CHANNEL_ORIGIN[0] + CHANNEL_DIM[0],
        CHANNEL_ORIGIN[1],
        CHANNEL_ORIGIN[1] + CHANNEL_DIM[1],
        mask_fn,
    )

def sample_solid_I_hr(batch_size):
    # High-res region: inside box
    x = np.random.uniform(BOX_ORIGIN[0], BOX_ORIGIN[0] + BOX_DIM[0], (batch_size, 1))
    y = np.random.uniform(BOX_ORIGIN[1], BOX_ORIGIN[1] + BOX_DIM[1], (batch_size, 1))
    return torch.tensor(np.concatenate([x, y], axis=1), dtype=torch.float32)

def sample_solid_II(batch_size):
    # Fin + base
    x = np.random.uniform(HEAT_SINK_BASE_ORIGIN[0], HEAT_SINK_BASE_ORIGIN[0] + HEAT_SINK_BASE_DIM[0], (batch_size, 1))
    y = np.random.uniform(HEAT_SINK_BASE_ORIGIN[1], HEAT_SINK_BASE_ORIGIN[1] + HEAT_SINK_BASE_DIM[1], (batch_size, 1))
    x2 = np.random.uniform(FIN_ORIGIN[0], FIN_ORIGIN[0] + FIN_DIM[0], (batch_size, 1))
    y2 = np.random.uniform(FIN_ORIGIN[1], FIN_ORIGIN[1] + FIN_DIM[1], (batch_size, 1))
    pts = np.concatenate([np.concatenate([x, y], axis=1), np.concatenate([x2, y2], axis=1)], axis=0)
    return torch.tensor(pts[:batch_size], dtype=torch.float32)

def sample_inlet(batch_size):
    y = np.random.uniform(CHANNEL_ORIGIN[1], CHANNEL_ORIGIN[1] + CHANNEL_DIM[1], (batch_size, 1))
    x = np.full((batch_size, 1), CHANNEL_ORIGIN[0])
    return torch.tensor(np.concatenate([x, y], axis=1), dtype=torch.float32)

def sample_outlet(batch_size):
    y = np.random.uniform(CHANNEL_ORIGIN[1], CHANNEL_ORIGIN[1] + CHANNEL_DIM[1], (batch_size, 1))
    x = np.full((batch_size, 1), CHANNEL_ORIGIN[0] + CHANNEL_DIM[0])
    return torch.tensor(np.concatenate([x, y], axis=1), dtype=torch.float32)

def sample_walls(batch_size):
    n = batch_size // 2
    x1 = np.random.uniform(CHANNEL_ORIGIN[0], CHANNEL_ORIGIN[0] + CHANNEL_DIM[0], (n, 1))
    y1 = np.full((n, 1), CHANNEL_ORIGIN[1])
    x2 = np.random.uniform(CHANNEL_ORIGIN[0], CHANNEL_ORIGIN[0] + CHANNEL_DIM[0], (n, 1))
    y2 = np.full((n, 1), CHANNEL_ORIGIN[1] + CHANNEL_DIM[1])
    pts = np.concatenate([np.concatenate([x1, y1], axis=1), np.concatenate([x2, y2], axis=1)], axis=0)
    return torch.tensor(pts[:batch_size], dtype=torch.float32)

def sample_chip_walls(batch_size):
    n = batch_size // 2
    x1 = np.full((n, 1), HEAT_SINK_BASE_ORIGIN[0])
    y1 = np.random.uniform(HEAT_SINK_BASE_ORIGIN[1], HEAT_SINK_BASE_ORIGIN[1] + HEAT_SINK_BASE_DIM[1], (n, 1))
    x2 = np.full((n, 1), HEAT_SINK_BASE_ORIGIN[0] + HEAT_SINK_BASE_DIM[0])
    y2 = np.random.uniform(HEAT_SINK_BASE_ORIGIN[1], HEAT_SINK_BASE_ORIGIN[1] + HEAT_SINK_BASE_DIM[1], (n, 1))
    pts = np.concatenate([np.concatenate([x1, y1], axis=1), np.concatenate([x2, y2], axis=1)], axis=0)
    return torch.tensor(pts[:batch_size], dtype=torch.float32)

class SolidINet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 1)  # theta_I
        )
    def forward(self, x):
        return self.net(x)

class SolidIINet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 1)  # theta_II
        )
    def forward(self, x):
        return self.net(x)

def diffusion_residual(xy, net):
    xy.requires_grad_(True)
    theta = net(xy)
    grads = torch.autograd.grad(theta.sum(), xy, create_graph=True)[0]
    theta_x = grads[:, 0:1]
    theta_y = grads[:, 1:2]
    theta_xx = torch.autograd.grad(theta_x.sum(), xy, create_graph=True)[0][:, 0:1]
    theta_yy = torch.autograd.grad(theta_y.sum(), xy, create_graph=True)[0][:, 1:2]
    res = theta_xx + theta_yy
    return res

def train(max_steps):
    net_I = SolidINet()
    net_II = SolidIINet()
    optimizer = optim.Adam(list(net_I.parameters()) + list(net_II.parameters()), lr=LEARNING_RATE)
    for step in range(max_steps):
        # Solid I interior (lr)
        xy_lr = sample_solid_I_lr(BATCH_SIZES['interior_lr'])
        res_lr = diffusion_residual(xy_lr, net_I)
        loss_lr = (res_lr**2).mean()
        # Solid I interior (hr)
        xy_hr = sample_solid_I_hr(BATCH_SIZES['interior_hr'])
        res_hr = diffusion_residual(xy_hr, net_I)
        loss_hr = (res_hr**2).mean()
        # Solid II interior
        xy_II = sample_solid_II(BATCH_SIZES['interiorS'])
        res_II = diffusion_residual(xy_II, net_II)
        loss_II = (res_II**2).mean()
        # Inlet BC (Dirichlet)
        xy_inlet = sample_inlet(BATCH_SIZES['inlet'])
        theta_inlet = net_I(xy_inlet)
        loss_inlet = ((theta_inlet - INLET_TEMP)**2).mean()
        # Outlet BC (Neumann)
        xy_outlet = sample_outlet(BATCH_SIZES['outlet'])
        xy_outlet.requires_grad_(True)
        theta_outlet = net_I(xy_outlet)
        grads = torch.autograd.grad(theta_outlet.sum(), xy_outlet, create_graph=True)[0]
        loss_outlet = (grads[:, 0:1]**2).mean()
        # Channel walls (Neumann)
        xy_walls = sample_walls(BATCH_SIZES['walls'])
        xy_walls.requires_grad_(True)
        theta_walls = net_I(xy_walls)
        grads = torch.autograd.grad(theta_walls.sum(), xy_walls, create_graph=True)[0]
        loss_walls = (grads[:, 1:2]**2).mean()
        # Chip walls (Neumann for solid II)
        xy_chip_walls = sample_chip_walls(BATCH_SIZES['chip_walls'])
        xy_chip_walls.requires_grad_(True)
        theta_chip = net_II(xy_chip_walls)
        grads = torch.autograd.grad(theta_chip.sum(), xy_chip_walls, create_graph=True)[0]
        loss_chip_walls = (grads[:, 0:1]**2 + grads[:, 1:2]**2).mean()
        # Interface (Dirichlet continuity)
        xy_interface = sample_chip_walls(BATCH_SIZES['interface'])
        theta_I = net_I(xy_interface)
        theta_II = net_II(xy_interface)
        loss_interface = ((theta_I - theta_II)**2).mean()
        # Heat source (Neumann for solid II, bottom of chip)
        x = np.random.uniform(SOURCE_ORIGIN[0], SOURCE_ORIGIN[0] + SOURCE_DIM[0], (BATCH_SIZES['heat_source'], 1))
        y = np.full((BATCH_SIZES['heat_source'], 1), SOURCE_ORIGIN[1])
        xy_heat = torch.tensor(np.concatenate([x, y], axis=1), dtype=torch.float32)
        xy_heat.requires_grad_(True)
        theta_heat = net_II(xy_heat)
        grads = torch.autograd.grad(theta_heat.sum(), xy_heat, create_graph=True)[0]
        loss_heat_source = ((grads[:, 1:2] - SOURCE_GRAD)**2).mean()
        # Total loss
        loss = loss_lr + loss_hr + loss_II + loss_inlet + loss_outlet + loss_walls + loss_chip_walls + loss_interface + loss_heat_source
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 1000 == 0:
            print(f"Step {step}: loss={loss.item():.6f}")
    torch.save({'net_I': net_I.state_dict(), 'net_II': net_II.state_dict()}, "chip2d_solid_solid_pinn.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_steps", type=int, default=MAX_STEPS)
    args = parser.parse_args()
    train(args.max_steps)
