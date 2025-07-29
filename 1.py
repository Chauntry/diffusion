import torch
import torch.nn as nn


class DIT(nn.Module):

    def forward(self, x, t):

        return x
    

model = DIT()
real_model = DIT()
fake_model = DIT()


def generate_new(model, x, num_steps, timesteps, eta):
    

    noise_latent_list = []

    latent = x

    print(timesteps.shape, timesteps[0].shape)

    for idx in range(num_steps):

        t = timesteps[idx]


        sigma = 1 - t

        print(t, t.shape, sigma.shape)

        sigma = sigma[:, None, None, None]

        model_v = model(latent, t)

        x0 = latent - (1 - sigma) * model_v
        x0_eps = eta * x0 + (1 - eta**2) ** 0.5 * torch.randn(xt_g.shape)

        x1 = latent + sigma * model_v


        t = timesteps[idx + 1]
        sigma_new = 1 - t
        sigma_new = sigma_new[:, None, None, None]

        latent = x0_eps * sigma_new + x1 * (1 - sigma_new)

        noise_latent_list.append(latent)

    return noise_latent_list

def predict(model, xt_g, t_g):


    model_v = model(xt_g, t_g)


    return xt_g


def add_noise(x, noise, t1, t2):

    sigma1 = 1 - t1
    sigma2 = 1 - t2

    sample = x * (1 - sigma2) / (1 - sigma1)

    beta = sigma2 ** 2 - sigma1 * (1 - sigma2) / (1 - sigma1)

    beta = beta ** 0.5

    return sample + beta * noise




bs = 2
num_steps = 8
x = torch.randn(bs, 30, 90, 160)

timesteps = torch.linspace(0, 1, num_steps+1)

time_shifting_factor = 4
# timesteps = shift * timesteps / (1 + (shift - 1) * timesteps)
timesteps = timesteps / (timesteps + time_shifting_factor - time_shifting_factor * timesteps)

# tensor([0.0000, 0.3636, 0.5714, 0.7059, 0.8000, 0.8696, 0.9231, 0.9655, 1.0000])

print(timesteps)

eta = 0.8

noise = torch.randn(x.shape)

with torch.no_grad():
    noise_latent_list = generate_new(model, x, num_steps, timesteps, eta)


t_idx = torch.randint(0, num_steps, (bs,))
xt_g = torch.randn(x.shape)

for i in range(bs):
    xt_g[i] = noise_latent_list[t_idx[i]]

t_g = timesteps[t_idx]
t_mid = timesteps[t_idx + 1]

t = torch.randn(t_mid.shape) * t_mid
# print(xt_g.shape, t_g)

x0, x1 = predict(model, xt_g, t_g)

x0_eps = eta * x0 + (1 - eta**2) ** 0.5 * torch.randn(xt_g.shape)

sigma_mid = 1 - t_mid
sigma_mid = sigma_mid[:, None, None, None]

x_mid = sigma_mid * x0_eps + (1 - sigma_mid) * x1

x_t = add_noise(x_mid, torch.randn(x.shape), t_mid, t)

v_fake = fake_model(x_t, t)
v_target = (x1 - x_t) / sigma_mid

loss_fake = torch.mean((v_fake - v_target)**2)


# ===========================================



t_idx = torch.randint(0, num_steps, (bs,))
xt_g = torch.randn(x.shape)

for i in range(bs):
    xt_g[i] = noise_latent_list[t_idx[i]]

t_g = timesteps[t_idx]
t_mid = timesteps[t_idx + 1]

t = torch.randn(t_mid.shape) * t_mid
# print(xt_g.shape, t_g)

x0, x1 = predict(model, xt_g, t_g)

x0_eps = eta * x0 + (1 - eta**2) ** 0.5 * torch.randn(xt_g.shape)

sigma_mid = 1 - t_mid
sigma_mid = sigma_mid[:, None, None, None]


x_mid = sigma_mid * x0_eps + (1 - sigma_mid) * x1

x_t = add_noise(x_mid, torch.randn(x.shape), t_mid, t)

with torch.no_grad():
    v_fake = predict(fake_model, x_t, t)
    v_real = predict(real_model, x_t, t)
    v_revisit = x1 + v_fake - v_real

huber_c = 0.15

loss = ((x1 - x_t) ** 2 + huber_c ** 2) ** 0.5 - huber_c

