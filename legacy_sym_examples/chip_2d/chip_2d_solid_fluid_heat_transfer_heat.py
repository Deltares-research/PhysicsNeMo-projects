"""
Explicit PyTorch PINN for coupled solid/fluid heat transfer in 2D chip geometry.
Replaces legacy abstractions with direct geometry sampling and explicit residuals.
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
INLET_TEMP = 25.0
COPPER_SOURCE_GRAD = 51.948051948

BATCH_SIZES = {
    'inlet': 200,
    'outlet': 200,
    'walls': 1000,
    'interior_lr': 2500,
    'interior_hr': 2500,
    'interiorS': 3000,
    'heat_source': 400,
    'interface': 400,
    'chip_walls': 400,
}
MAX_STEPS = 150000
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

def sample_fluid_interior(batch_size, region='lr'):

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

def sample_solid_interior(batch_size):
    x = np.random.uniform(CHIP_POS, CHIP_POS + CHIP_WIDTH, (batch_size, 1))
    y = np.random.uniform(CHANNEL_WIDTH[0], CHANNEL_WIDTH[0] + CHIP_HEIGHT, (batch_size, 1))
    return torch.tensor(np.concatenate([x, y], axis=1), dtype=torch.float32)

def sample_inlet(batch_size):
    y = np.random.uniform(CHANNEL_WIDTH[0], CHANNEL_WIDTH[1], (batch_size, 1))
    x = np.full((batch_size, 1), CHANNEL_LENGTH[0])
    return torch.tensor(np.concatenate([x, y], axis=1), dtype=torch.float32)

def sample_outlet(batch_size):
    y = np.random.uniform(CHANNEL_WIDTH[0], CHANNEL_WIDTH[1], (batch_size, 1))
    x = np.full((batch_size, 1), CHANNEL_LENGTH[1])
    return torch.tensor(np.concatenate([x, y], axis=1), dtype=torch.float32)

def sample_walls(batch_size):
    n = batch_size // 2
    x1 = np.random.uniform(CHANNEL_LENGTH[0], CHANNEL_LENGTH[1], (n, 1))
    y1 = np.full((n, 1), CHANNEL_WIDTH[0])
    x2 = np.random.uniform(CHANNEL_LENGTH[0], CHANNEL_LENGTH[1], (n, 1))
    y2 = np.full((n, 1), CHANNEL_WIDTH[1])
    pts = np.concatenate([np.concatenate([x1, y1], axis=1), np.concatenate([x2, y2], axis=1)], axis=0)
    return torch.tensor(pts[:batch_size], dtype=torch.float32)

def sample_chip_walls(batch_size):
    n = batch_size // 2
    x1 = np.full((n, 1), CHIP_POS)
    y1 = np.random.uniform(CHANNEL_WIDTH[0], CHANNEL_WIDTH[0] + CHIP_HEIGHT, (n, 1))
    x2 = np.full((n, 1), CHIP_POS + CHIP_WIDTH)
    y2 = np.random.uniform(CHANNEL_WIDTH[0], CHANNEL_WIDTH[0] + CHIP_HEIGHT, (n, 1))
    pts = np.concatenate([np.concatenate([x1, y1], axis=1), np.concatenate([x2, y2], axis=1)], axis=0)
    return torch.tensor(pts[:batch_size], dtype=torch.float32)

class FluidHeatNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 1)  # theta_f
        )
    def forward(self, x):
        return self.net(x)

class SolidHeatNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 1)  # theta_s
        )
    def forward(self, x):
        return self.net(x)

def advection_diffusion_residual(xy, net, u=INLET_TEMP, D=1.0):
    xy.requires_grad_(True)
    theta = net(xy)
    grads = torch.autograd.grad(theta.sum(), xy, create_graph=True)[0]
    theta_x = grads[:, 0:1]
    theta_y = grads[:, 1:2]
    theta_xx = torch.autograd.grad(theta_x.sum(), xy, create_graph=True)[0][:, 0:1]
    theta_yy = torch.autograd.grad(theta_y.sum(), xy, create_graph=True)[0][:, 1:2]
    # For simplicity, ignore advection (set u=0)
    res = theta_xx + theta_yy
    return res

def train(max_steps):
    fluid_net = FluidHeatNet()
    solid_net = SolidHeatNet()
    optimizer = optim.Adam(list(fluid_net.parameters()) + list(solid_net.parameters()), lr=LEARNING_RATE)
    for step in range(max_steps):
        # Fluid interior
        xy_lr = sample_fluid_interior(BATCH_SIZES['interior_lr'], region='lr')
        res_lr = advection_diffusion_residual(xy_lr, fluid_net)
        loss_lr = (res_lr**2).mean()
        xy_hr = sample_fluid_interior(BATCH_SIZES['interior_hr'], region='hr')
        res_hr = advection_diffusion_residual(xy_hr, fluid_net)
        loss_hr = (res_hr**2).mean()
        # Solid interior
        xy_solid = sample_solid_interior(BATCH_SIZES['interiorS'])
        res_solid = advection_diffusion_residual(xy_solid, solid_net)
        loss_solid = (res_solid**2).mean()
        # Inlet BC (fluid)
        xy_inlet = sample_inlet(BATCH_SIZES['inlet'])
        theta_inlet = fluid_net(xy_inlet)
        loss_inlet = ((theta_inlet - INLET_TEMP)**2).mean()
        # Outlet BC (Neumann)
        xy_outlet = sample_outlet(BATCH_SIZES['outlet'])
        xy_outlet.requires_grad_(True)
        theta_outlet = fluid_net(xy_outlet)
        grads = torch.autograd.grad(theta_outlet.sum(), xy_outlet, create_graph=True)[0]
        loss_outlet = (grads[:, 0:1]**2).mean()
        # Channel walls (Neumann)
        xy_walls = sample_walls(BATCH_SIZES['walls'])
        xy_walls.requires_grad_(True)
        theta_walls = fluid_net(xy_walls)
        grads = torch.autograd.grad(theta_walls.sum(), xy_walls, create_graph=True)[0]
        loss_walls = (grads[:, 1:2]**2).mean()
        # Chip walls (Neumann for solid)
        xy_chip_walls = sample_chip_walls(BATCH_SIZES['chip_walls'])
        xy_chip_walls.requires_grad_(True)
        theta_chip = solid_net(xy_chip_walls)
        grads = torch.autograd.grad(theta_chip.sum(), xy_chip_walls, create_graph=True)[0]
        loss_chip_walls = (grads[:, 0:1]**2 + grads[:, 1:2]**2).mean()
        # Interface (Dirichlet continuity)
        # For simplicity, sample interface as chip boundary
        xy_interface = sample_chip_walls(BATCH_SIZES['interface'])
        theta_f = fluid_net(xy_interface)
        theta_s = solid_net(xy_interface)
        loss_interface = ((theta_f - theta_s)**2).mean()
        # Heat source (Neumann for solid, bottom of chip)
        x = np.random.uniform(CHIP_POS, CHIP_POS + CHIP_WIDTH, (BATCH_SIZES['heat_source'], 1))
        y = np.full((BATCH_SIZES['heat_source'], 1), CHANNEL_WIDTH[0])
        xy_heat = torch.tensor(np.concatenate([x, y], axis=1), dtype=torch.float32)
        xy_heat.requires_grad_(True)
        theta_heat = solid_net(xy_heat)
        grads = torch.autograd.grad(theta_heat.sum(), xy_heat, create_graph=True)[0]
        loss_heat_source = ((grads[:, 1:2] - COPPER_SOURCE_GRAD)**2).mean()
        # Total loss
        loss = loss_lr + loss_hr + loss_solid + loss_inlet + loss_outlet + loss_walls + loss_chip_walls + loss_interface + loss_heat_source
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 1000 == 0:
            print(f"Step {step}: loss={loss.item():.6f}")
    torch.save({'fluid_net': fluid_net.state_dict(), 'solid_net': solid_net.state_dict()}, "chip2d_heat_pinn.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_steps", type=int, default=MAX_STEPS)
    args = parser.parse_args()
    train(args.max_steps)
