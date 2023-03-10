import torch.nn as nn
import torch.nn.functional as F
from utils.diffusion_utils import *
from utils.resnet import *


class ConditionalLinear(nn.Module):
    def __init__(self, num_in, num_out, n_steps):
        super(ConditionalLinear, self).__init__()
        self.num_out = num_out
        self.lin = nn.Linear(num_in, num_out)
        self.embed = nn.Embedding(n_steps, num_out)
        self.embed.weight.data.uniform_()

    def forward(self, x, t):
        out = self.lin(x)
        gamma = self.embed(t)
        out = gamma.view(-1, self.num_out) * out
        return out


class ConditionalModel(nn.Module):
    def __init__(self, n_steps, y_dim=10, fp_dim=128, feature_dim=4096, guidance=True, encoder_type='resnet34'):
        super(ConditionalModel, self).__init__()
        n_steps = n_steps + 1
        self.y_dim = y_dim
        self.guidance = guidance
        self.encoder_type = encoder_type

        if encoder_type == 'resnet34':
            self.encoder_x = resnet34(num_input_channels=3, num_classes=feature_dim)
        elif encoder_type == 'resnet18':
            self.encoder_x = resnet18(num_input_channels=3, num_classes=feature_dim)
        elif encoder_type == 'resnet101':
            self.encoder_x = resnet101(num_input_channels=3, num_classes=feature_dim)
        elif encoder_type == 'resnet50':
            self.encoder_x = resnet50(num_input_channels=3, num_classes=feature_dim)
        elif encoder_type == 'linear':
            self.encoder_x = nn.Sequential(
            nn.Linear(3072, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.Softplus(),
            nn.Linear(feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.Softplus(),
            nn.Linear(feature_dim, feature_dim)
        )
        else:
            raise Exception("ResNet type should be one of [18, 34, 50, 101, 152]")

        # batch norm layer
        self.norm = nn.BatchNorm1d(feature_dim)

        # Unet
        if self.guidance:
            self.lin1 = ConditionalLinear(y_dim + fp_dim, feature_dim, n_steps)
        else:
            self.lin1 = ConditionalLinear(y_dim, feature_dim, n_steps)

        self.unetnorm1 = nn.BatchNorm1d(feature_dim)
        self.lin2 = ConditionalLinear(feature_dim, feature_dim, n_steps)
        self.unetnorm2 = nn.BatchNorm1d(feature_dim)
        self.lin3 = ConditionalLinear(feature_dim, feature_dim, n_steps)
        self.unetnorm3 = nn.BatchNorm1d(feature_dim)
        self.lin4 = nn.Linear(feature_dim, y_dim)

    def forward(self, x, y, t, fp_x=None):

        if self.encoder_type == 'linear':
            x_flat = torch.flatten(x, 1)
            x_embed = self.encoder_x(x_flat)
        else:
            x_embed = self.encoder_x(x)

        x_embed = self.norm(x_embed)

        if self.guidance:
            y = torch.cat([y, fp_x], dim=-1)

        y = self.lin1(y, t)
        y = self.unetnorm1(y)
        y = F.softplus(y)
        y = x_embed * y
        y = self.lin2(y, t)
        y = self.unetnorm2(y)
        y = F.softplus(y)
        y = self.lin3(y, t)
        y = self.unetnorm3(y)
        y = F.softplus(y)
        return self.lin4(y)


class Diffusion(nn.Module):
    def __init__(self, fp_encoder, num_timesteps=1000, n_class=10, fp_dim=512, device='cuda', beta_schedule='linear',
                 feature_dim=2048, encoder_type='resnet34', ddim_num_steps=10):
        super().__init__()
        self.device = device
        self.num_timesteps = num_timesteps
        self.n_class = n_class
        betas = make_beta_schedule(schedule=beta_schedule, num_timesteps=self.num_timesteps, start=0.0001, end=0.02)
        betas = self.betas = betas.float().to(self.device)
        self.betas_sqrt = torch.sqrt(betas)
        alphas = 1.0 - betas
        self.alphas = alphas
        self.one_minus_betas_sqrt = torch.sqrt(alphas)
        self.alphas_cumprod = alphas.cumprod(dim=0)
        self.alphas_bar_sqrt = torch.sqrt(self.alphas_cumprod)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - self.alphas_cumprod)
        alphas_cumprod_prev = torch.cat([torch.ones(1).to(self.device), self.alphas_cumprod[:-1]], dim=0)
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.posterior_mean_coeff_1 = (betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.posterior_mean_coeff_2 = (torch.sqrt(alphas) * (1 - alphas_cumprod_prev) / (1 - self.alphas_cumprod))
        posterior_variance = (betas * (1.0 - alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.posterior_variance = posterior_variance
        self.logvar = betas.log()
        self.fp_dim = fp_dim

        self.fp_encoder = fp_encoder
        self.fp_encoder.eval()
        self.model = ConditionalModel(self.num_timesteps, y_dim=self.n_class, fp_dim=fp_dim,
                                      feature_dim=feature_dim, guidance=True, encoder_type=encoder_type).to(self.device)

        self.ddim_num_steps = ddim_num_steps
        self.make_ddim_schedule(ddim_num_steps)

    def make_ddim_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0.):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.num_timesteps)

        assert self.alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.device)

        self.register_buffer('sqrt_alphas_cumprod', to_torch(torch.sqrt(self.alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(torch.sqrt(1. - self.alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(torch.log(1. - self.alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(torch.sqrt(1. / self.alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(torch.sqrt(1. / self.alphas_cumprod - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=self.alphas_cumprod,
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', torch.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    def load_diffusion_net(self, net_state_dicts):
        self.model.load_state_dict(net_state_dicts[0])
        self.fp_encoder.load_state_dict(net_state_dicts[1])

    def forward_t(self, y_0_batch, x_batch, t, fp_x, fq_x=None):

        x_batch = x_batch.to(self.device)

        e = torch.randn_like(y_0_batch).to(y_0_batch.device)
        y_t_batch = q_sample(y_0_batch, self.alphas_bar_sqrt,
                             self.one_minus_alphas_bar_sqrt, t, noise=e, fq_x=fq_x)

        output = self.model(x_batch, y_t_batch, t, fp_x)

        return output, e

    def reverse(self, images, only_last_sample=True, stochastic=True, fq_x=None):

        images = images.to(self.device)
        with torch.no_grad():

            fp_x = self.fp_encoder(images)
            label_t_0 = p_sample_loop(self.model, images, fp_x,
                                      self.num_timesteps, self.alphas,
                                      self.one_minus_alphas_bar_sqrt,
                                      only_last_sample=only_last_sample, stochastic=stochastic, fq_x=fq_x)

        return label_t_0

    def reverse_ddim(self, images, stochastic=True, fq_x=None):

        images = images.to(self.device)
        with torch.no_grad():

            fp_x = self.fp_encoder(images)
            label_t_0 = ddim_sample_loop(self.model, images, fp_x, self.ddim_timesteps, self.ddim_alphas,
                                         self.ddim_alphas_prev, self.ddim_sigmas, stochastic=stochastic, fq_x=fq_x)

        return label_t_0

