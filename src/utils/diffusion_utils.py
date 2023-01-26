import math
import torch
import numpy as np
from tqdm import tqdm

def make_beta_schedule(schedule="linear", num_timesteps=1000, start=1e-5, end=1e-2):
    if schedule == "linear":
        betas = torch.linspace(start, end, num_timesteps)
    elif schedule == "const":
        betas = end * torch.ones(num_timesteps)
    elif schedule == "quad":
        betas = torch.linspace(start ** 0.5, end ** 0.5, num_timesteps) ** 2
    elif schedule == "jsd":
        betas = 1.0 / torch.linspace(num_timesteps, 1, num_timesteps)
    elif schedule == "sigmoid":
        betas = torch.linspace(-6, 6, num_timesteps)
        betas = torch.sigmoid(betas) * (end - start) + start
    elif schedule == "cosine" or schedule == "cosine_reverse":
        max_beta = 0.999
        cosine_s = 0.008
        betas = torch.tensor(
            [min(1 - (math.cos(((i + 1) / num_timesteps + cosine_s) / (1 + cosine_s) * math.pi / 2) ** 2) / (
                    math.cos((i / num_timesteps + cosine_s) / (1 + cosine_s) * math.pi / 2) ** 2), max_beta) for i in
             range(num_timesteps)])
    elif schedule == "cosine_anneal":
        betas = torch.tensor(
            [start + 0.5 * (end - start) * (1 - math.cos(t / (num_timesteps - 1) * math.pi)) for t in
             range(num_timesteps)])
    return betas


def extract(input, t, x):
    shape = x.shape
    out = torch.gather(input, 0, t.to(input.device))
    reshape = [t.shape[0]] + [1] * (len(shape) - 1)
    return out.reshape(*reshape)


# Forward functions
def q_sample(y, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, t, noise=None):

    if noise is None:
        noise = torch.randn_like(y).to(y.device)
    sqrt_alpha_bar_t = extract(alphas_bar_sqrt, t, y)
    sqrt_one_minus_alpha_bar_t = extract(one_minus_alphas_bar_sqrt, t, y)
    # q(y_t | y_0, x)
    y_t = sqrt_alpha_bar_t * y + sqrt_one_minus_alpha_bar_t * noise
    return y_t


# Reverse function -- sample y_{t-1} given y_t
def p_sample(model, x, y_t, y_0_clr, t, alphas, one_minus_alphas_bar_sqrt, stochastic=False):
    """
    Reverse diffusion process sampling -- one time step.
    y: sampled y at time step t, y_t.
    y_0_pred: prediction of classifier model.
    y_T_mean: mean of prior distribution at timestep T.
    We replace y_0_hat with y_T_mean in the forward process posterior mean computation, emphasizing that 
        guidance model prediction y_0_hat = f_phi(x) is part of the input to eps_theta network, while 
        in paper we also choose to set the prior mean at timestep T y_T_mean = f_phi(x).
    """
    device = next(model.parameters()).device
    z = stochastic * torch.randn_like(y_t)
    t = torch.tensor([t]).to(device)
    alpha_t = extract(alphas, t, y_t)
    sqrt_one_minus_alpha_bar_t = extract(one_minus_alphas_bar_sqrt, t, y_t)
    sqrt_one_minus_alpha_bar_t_m_1 = extract(one_minus_alphas_bar_sqrt, t - 1, y_t)
    sqrt_alpha_bar_t = (1 - sqrt_one_minus_alpha_bar_t.square()).sqrt()
    sqrt_alpha_bar_t_m_1 = (1 - sqrt_one_minus_alpha_bar_t_m_1.square()).sqrt()
    # y_t_m_1 posterior mean component coefficients
    gamma_0 = (1 - alpha_t) * sqrt_alpha_bar_t_m_1 / (sqrt_one_minus_alpha_bar_t.square())
    gamma_1 = (sqrt_one_minus_alpha_bar_t_m_1.square()) * (alpha_t.sqrt()) / (sqrt_one_minus_alpha_bar_t.square())
    eps_theta = model(x, y_t, t, y_0_clr).to(device).detach()
    # y_0 reparameterization
    y_0_reparam = 1 / sqrt_alpha_bar_t * (
            y_t - eps_theta * sqrt_one_minus_alpha_bar_t)
    # posterior mean
    y_t_m_1_hat = gamma_0 * y_0_reparam + gamma_1 * y_t
    # posterior variance
    beta_t_hat = (sqrt_one_minus_alpha_bar_t_m_1.square()) / (sqrt_one_minus_alpha_bar_t.square()) * (1 - alpha_t)
    y_t_m_1 = y_t_m_1_hat.to(device) + beta_t_hat.sqrt().to(device) * z.to(device)
    return y_t_m_1


# Reverse function -- sample y_0 given y_1
def p_sample_t_1to0(model, x, y_t, y_0_clr, one_minus_alphas_bar_sqrt):
    device = next(model.parameters()).device
    t = torch.tensor([0]).to(device)  # corresponding to timestep 1 (i.e., t=1 in diffusion models)
    sqrt_one_minus_alpha_bar_t = extract(one_minus_alphas_bar_sqrt, t, y_t)
    sqrt_alpha_bar_t = (1 - sqrt_one_minus_alpha_bar_t.square()).sqrt()
    eps_theta = model(x, y_t, t, y_0_clr).to(device).detach()
    # y_0 reparameterization
    y_0_reparam = 1 / sqrt_alpha_bar_t * (
            y_t - eps_theta * sqrt_one_minus_alpha_bar_t)
    y_t_m_1 = y_0_reparam.to(device)
    return y_t_m_1


def y_0_reparam(model, x, y_t, y_0_clr, t, one_minus_alphas_bar_sqrt):
    """
    Obtain y_0 reparameterization from q(y_t | y_0), in which noise term is the eps_theta prediction.
    Algorithm 2 Line 4 in paper.
    """
    device = next(model.parameters()).device
    sqrt_one_minus_alpha_bar_t = extract(one_minus_alphas_bar_sqrt, t, y_t)
    sqrt_alpha_bar_t = (1 - sqrt_one_minus_alpha_bar_t.square()).sqrt()
    eps_theta = model(x, y_t, t, y_0_clr).to(device).detach()
    # y_0 reparameterization
    y_0_reparam = 1 / sqrt_alpha_bar_t * (
            y_t - eps_theta * sqrt_one_minus_alpha_bar_t).to(device)
    return y_0_reparam


def p_sample_loop(model, x, y_0_clr, n_steps, alphas, one_minus_alphas_bar_sqrt,
                  only_last_sample=True, stochastic=True):
    num_t, y_p_seq = None, None
    device = next(model.parameters()).device
    cur_y = stochastic * torch.randn_like(y_0_clr).to(device)

    if only_last_sample:
        num_t = 1
    else:
        # y_p_seq = [cur_y]
        y_p_seq = torch.zeros([cur_y.shape[0], cur_y.shape[1], n_steps+1]).to(device)
        y_p_seq[:, :, n_steps] = cur_y
    for t in reversed(range(1, n_steps)):
        y_t = cur_y
        cur_y = p_sample(model, x, y_t, y_0_clr, t, alphas, one_minus_alphas_bar_sqrt, stochastic=stochastic)  # y_{t-1}
        if only_last_sample:
            num_t += 1
        else:
            # y_p_seq.append(cur_y)
            y_p_seq[:, :, t] = cur_y
    if only_last_sample:
        assert num_t == n_steps
        y_0 = p_sample_t_1to0(model, x, cur_y, y_0_clr, one_minus_alphas_bar_sqrt)
        return y_0
    else:
        # assert len(y_p_seq) == n_steps
        y_0 = p_sample_t_1to0(model, x, y_p_seq[:, :, 1], y_0_clr, one_minus_alphas_bar_sqrt)
        # y_p_seq.append(y_0)
        y_p_seq[:, :, 0] = y_0
        return y_p_seq


def make_ddim_timesteps(ddim_discr_method, num_ddim_timesteps, num_ddpm_timesteps):
    if ddim_discr_method == 'uniform':
        c = num_ddpm_timesteps // num_ddim_timesteps
        ddim_timesteps = np.asarray(list(range(0, num_ddpm_timesteps, c)))
    elif ddim_discr_method == 'quad':
        ddim_timesteps = ((np.linspace(0, np.sqrt(num_ddpm_timesteps * .8), num_ddim_timesteps)) ** 2).astype(int)
    else:
        raise NotImplementedError(f'There is no ddim discretization method called "{ddim_discr_method}"')

    # assert ddim_timesteps.shape[0] == num_ddim_timesteps
    # add one to get the final alpha values right (the ones from first scale to data during sampling)
    steps_out = ddim_timesteps + 1
    # print(f'Selected timesteps for ddim sampler: {steps_out}')

    return steps_out


def make_ddim_sampling_parameters(alphacums, ddim_timesteps, eta):
    # select alphas for computing the variance schedule
    device = alphacums.device
    alphas = alphacums[ddim_timesteps]
    alphas_prev = torch.tensor([alphacums[0]] + alphacums[ddim_timesteps[:-1]].tolist()).to(device)

    # according the the formula provided in https://arxiv.org/abs/2010.02502
    sigmas = eta * torch.sqrt((1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev))
    # print(f'Selected alphas for ddim sampler: a_t: {alphas}; a_(t-1): {alphas_prev}')
    # print(f'For the chosen value of eta, which is {eta}, '
    #       f'this results in the following sigma_t schedule for ddim sampler {sigmas}')
    return sigmas, alphas, alphas_prev


def ddim_sample_loop(model, x, y_0_clr, timesteps, ddim_alphas, ddim_alphas_prev, ddim_sigmas, stochastic=True):
    device = next(model.parameters()).device
    batch_size = x.shape[0]

    y_t = stochastic * torch.randn_like(torch.zeros([batch_size, model.y_dim])).to(device)

    # intermediates = {'y_inter': [y_t], 'pred_y0': [y_t]}
    time_range = np.flip(timesteps)
    total_steps = timesteps.shape[0]
    # print(f"Running DDIM Sampling with {total_steps} timesteps")

    for i, step in enumerate(time_range):
        index = total_steps - i - 1
        t = torch.full((batch_size,), step, device=device, dtype=torch.long)

        y_t, pred_y0 = ddim_sample_step(model, x, y_t, y_0_clr, t, index, ddim_alphas, ddim_alphas_prev, ddim_sigmas)

        # intermediates['y_inter'].append(y_t)
        # intermediates['pred_y0'].append(pred_y0)

    return y_t


def ddim_sample_step(model, x, y_t, y_0_clr, t, index, ddim_alphas, ddim_alphas_prev, ddim_sigmas):
    batch_size = x.shape[0]
    device = next(model.parameters()).device
    e_t = model(x, y_t, t, y_0_clr).to(device).detach()

    sqrt_one_minus_alphas = torch.sqrt(1. - ddim_alphas)
    # select parameters corresponding to the currently considered timestep
    a_t = torch.full([batch_size, 1], ddim_alphas[index], device=device)
    a_prev = torch.full([batch_size, 1], ddim_alphas_prev[index], device=device)
    sigma_t = torch.full([batch_size, 1], ddim_sigmas[index], device=device)
    sqrt_one_minus_at = torch.full([batch_size, 1], sqrt_one_minus_alphas[index], device=device)

    # current prediction for x_0
    pred_x0 = (y_t - sqrt_one_minus_at * e_t) / a_t.sqrt()

    # direction pointing to x_t
    dir_yt = (1. - a_prev - sigma_t ** 2).sqrt() * e_t
    noise = sigma_t * torch.randn_like(y_t).to(device)

    x_prev = a_prev.sqrt() * pred_x0 + dir_yt + noise
    return x_prev, pred_x0


if __name__ == "__main__":

    x = torch.ones([10])