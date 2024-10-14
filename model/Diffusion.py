from torch import nn
import torch
import torch.nn.functional as F
def gather(consts: torch.Tensor, t: torch.Tensor):
    """Gather consts for $t$ and reshape to feature map shape"""
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1, 1)

class Diffusion(nn.Module):
    def __init__(self,eps_model: nn.Module,n_steps: int,device: torch.device):
        super(Diffusion, self).__init__()
        self.eps_model = eps_model
        self.beta = torch.linspace(start=0.0001,end=0.02,steps=n_steps,dtype=torch.float32,device=device)
        self.alpha = 1 - self.beta

        self.alpha_bar = torch.cumprod(self.alpha,dim=0)
        self.n_steps = n_steps
        self.sigma2 = self.beta

    def p_sample(self,xt,t,device):
        # 计算x_{t-1}
        # 根据公式: x_{t-1} = 1 / sqrt(alpha_t) * ( xt - (1-alpha_t) / (sqrt(1-alpha_bar_t)) * eps_model() )
        # + sigama_t * noise (noise是一个高斯噪声)
        xt = xt.to(device)
        t = t.to(device)
        coef1 = 1 / torch.sqrt(gather(self.alpha,t))
        eps = self.eps_model(xt,t)
        coef2 = (xt - (1-gather(self.alpha,t)) / torch.sqrt(gather(self.alpha_bar,t) * eps ))
        noise = torch.randn(xt.shape, device=xt.device)
        add = torch.sqrt(gather(self.sigma2,t)) * noise
        # 返回的这玩意就是x_{t-1}
        return coef1 * coef2 + add
    def q_sample(self,x0,t,eps):
        # x0,t,噪声eps
        # xt = sqrt ( alpha_bar_t ) * x0 + sqrt( 1 - alpha_bar_t) * eps
        alpha_bar_t = gather(self.alpha_bar,t)
        xt = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1-alpha_bar_t) * eps
        return xt
    def loss(self,x0):
        # 随机采样一个整数t
        batch_size = x0.shape[0]
        t = torch.randint(0,self.n_steps,(batch_size,),device=x0.device)
        noise = torch.randn_like(x0,device=x0.device) # 随机采样一个高斯噪声
        xt = self.q_sample(x0,t,eps=noise) # 得到加噪后的图片
        # print(xt.device)
        # print(t.device)
        eps_theta = self.eps_model(xt,t)
        return F.mse_loss(noise,eps_theta)