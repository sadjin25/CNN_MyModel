import torch
import torch.optim as optim
import torchvision
from torchvision import utils
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import psutil
import os

TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE = 256
EPOCH_SIZE = 20
SEED = 245
T = 100
TRAIN_DATA_PATH = './CNN_MyModel/MyModel/dataPT'


# TODO : sinusoidal embedding
class TimeEmbedding(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim)
        )
    
    def forward(self, t):
        t = t.float().unsqueeze(1)
        return self.mlp(t)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 
                               kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 
                                kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.time_emb = nn.Linear(time_emb_dim, out_channels)

        self.shortcut = nn.Identity()
        self.shortcut_bn = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            self.shortcut_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x, t_emb):
        orig = self.shortcut(x)
        orig = self.shortcut_bn(orig)

        x = F.relu(self.bn1(self.conv1(x)))
        
        time_feat = self.time_emb(t_emb)
        time_feat = time_feat[:,:,None,None] # [B,D] => [B,D,1,1] why? for correctioness with [B,C,H,W]
        
        x = x + time_feat # WARN : DONT in-place addition(+= or .add_). auto grad err occurs.  

        x = self.bn2(self.conv2(x))

        return F.relu(x + orig)

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.res1 = ResidualBlock(in_channels, out_channels, time_emb_dim)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x, t_emb):
        feat = self.res1(x, t_emb)
        down = self.pool(feat)
        return feat, down

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels, time_emb_dim):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.res1 = ResidualBlock(out_channels + skip_channels, out_channels, time_emb_dim)

    def forward(self, x, skip, t_emb):
        x = self.up(x)
        # x.shape = [B,C,H,W], correction of convtranspose err 
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        
        x = torch.cat((x, skip), dim=1)
        x = self.res1(x, t_emb)
        return x

class DiffusionUNet(nn.Module):
    def __init__(self, in_channels=3, time_emb_dim = 100):
        super().__init__()

        self.time_emb = TimeEmbedding(time_emb_dim)

        # Encoder
        self.down1 = DownBlock(in_channels, 64, time_emb_dim)
        self.down2 = DownBlock(64, 128, time_emb_dim)

        # Bottleneck
        self.res1 = ResidualBlock(128, 256, time_emb_dim)
        self.dropout_res = nn.Dropout2d(p=0.3)

        # Decoder
        self.up2 = UpBlock(256, 128, 128, time_emb_dim)
        self.up1 = UpBlock(128, 64, 64, time_emb_dim)

        # Output
        self.out = nn.Conv2d(64, in_channels, kernel_size=1)

    def forward(self, x, t):
        t_emb = self.time_emb(t)
        feat1, x = self.down1(x, t_emb)
        feat2, x = self.down2(x, t_emb)

        x = self.res1(x, t_emb)
        x = self.dropout_res(x)

        x = self.up2(x, feat2, t_emb)
        x = self.up1(x, feat1, t_emb)

        return self.out(x) # pred noise

def LinearBetaSchedule(timesteps, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)

def PrepareNoiseSchedule(timesteps, device):
    betas = LinearBetaSchedule(timesteps).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    return betas, alphas, alphas_cumprod, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod

def QSample(x0, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=None):
    if noise is None:
        noise = torch.randn_like(x0)
    
    a = sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
    b = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)

    xt = a * x0 + b * noise
    return xt, noise

def TrainingStep(model, x0, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, T):
    B = x0.size(0)
    t = torch.randint(0, T, (B,), device=x0.device)
    
    xt, noise = QSample(x0, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)

    pred_noise = model(xt, t)
    loss = F.mse_loss(pred_noise, noise)

    return loss

@torch.no_grad()
def PSample(model, x, t, betas, alphas, alpha_cumprod):
    B = x.size(0)
    device = x.device

    t_batch = torch.full((B,), t, device=device, dtype=torch.long)

    beta_t = betas[t]
    alpha_t = alphas[t]
    alpha_cumprod_t = alpha_cumprod[t]

    pred_noise = model(x, t_batch)
    coef1 = 1.0 / torch.sqrt(alpha_t)
    coef2 = beta_t / torch.sqrt(1.0 - alpha_cumprod_t)

    mean = coef1 * (x - pred_noise*coef2) 

    if t > 0:
        noise = torch.randn_like(x)
        sigma = torch.sqrt(beta_t)
        x_prev = mean + sigma * noise
    else:
        x_prev = mean

    return x_prev
    
# img size is 32 for CIFAR10.
@torch.no_grad()
def GenerateSamples(model, device, img_size=32, T=100, channels=3, num_samples=4):
    model.eval()
    
    betas, alphas, alpha_cumprod, _, _ = PrepareNoiseSchedule(T, device)

    x = torch.randn(num_samples, channels, img_size, img_size, device=device)

    for t in reversed(range(T)):
        x = PSample(model, x, t, betas, alphas, alpha_cumprod)

    return x

def denorm(x):
    # [-1, 1] -> [0, 1]
    return (x.clamp(-1, 1) + 1) / 2

def SaveSamples(model, device, epoch, save_dir, T, num_samples):
    model.eval()

    samples = GenerateSamples(
        model=model,
        device=device,
        img_size=32,
        T=T,
        channels=3,
        num_samples=num_samples
    )

    samples = denorm(samples)

    path = os.path.join(save_dir, f"sample_epoch_{epoch:03d}.png")
    utils.save_image(samples, path, nrow=4)
    
    print(f"saved samples: {path}")

@torch.no_grad()
def EvaluateLoss(model, val_loader, device, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, T):
    model.eval()
    total_loss = 0.0
    total_count = 0

    for x0, _ in val_loader:
        x0 = x0.to(device)
        B = x0.size(0)

        t = torch.randint(0, T, (B,), device=device)
        xt, noise = QSample(x0, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)

        pred_noise = model(xt, t)
        loss = F.mse_loss(pred_noise, noise, reduction='sum')

        total_loss += loss.item()
        total_count += noise.numel()

    return total_loss / total_count


def TrainDiffusion(epoches = EPOCH_SIZE, lr=2e-4, T=T):
    net = DiffusionUNet().to(DEVICE)
            
    # weight decay=1e-4 ?
    optimizer = optim.AdamW(net.parameters(), lr=lr)
    betas, alphas, alphas_cumprod, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod = PrepareNoiseSchedule(
        T,
        device=DEVICE
    )
   
    process = psutil.Process(os.getpid())
    peak_ram = 0.0

    for epoch in range(epoches):
        net.train() # DONT REMOVE IT. EvaluateLoss() in this code uses model.eval()  

        delta_time = time.perf_counter()
        total_loss = 0.0
        train_cases = 0

        for inputs, _ in trainloader:
            inputs = inputs.to(DEVICE, non_blocking=True)

            loss = TrainingStep(
                model = net,
                x0 = inputs,
                sqrt_alphas_cumprod=sqrt_alphas_cumprod,
                sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
                T=T
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            train_cases += inputs.size(0)
            
            ram = process.memory_info().rss
            peak_ram = max(peak_ram, ram)

        delta_time = time.perf_counter() - delta_time
       
        val_loss = EvaluateLoss(
        model=net,
        val_loader=testloader,
        device=DEVICE,
        sqrt_alphas_cumprod=sqrt_alphas_cumprod,
        sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
        T=T
        )

        print(f'FINISHED EPOCH[{epoch+1}] TRAIN, ELAPSED TIME : [{delta_time*1000:.3f}] ms')
        print(f'THROUGHPUT : [{train_cases/delta_time:.3f}]')
        print(f"[EPOCH {epoch+1}] TRAIN LOSS: {total_loss/train_cases:.3f}, VAL LOSS: {val_loss:.3f}")
        # rss output is byte size.
        print(f"PEAK RAM: {peak_ram / (1024**2):.2f} MB")
        
        SaveSamples(model=net, device=DEVICE, epoch=epoch+1, save_dir=TRAIN_DATA_PATH, T=T, num_samples=2)
        ckpt_path = os.path.join(TRAIN_DATA_PATH, "last.pt")
        
        torch.save({
            "model": net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "T": T,
        }, ckpt_path)

    return net

@torch.no_grad()
def LoadModelAndGenerate(ckpt_path=TRAIN_DATA_PATH + "/last.pt", num_samples=4):
    model = DiffusionUNet().to(DEVICE)

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()

    T_loaded = ckpt.get("T", T) # if ckpt has T, use it. else use const T.

    samples = GenerateSamples(
        model=model,
        device=DEVICE,
        img_size=32,
        T=T_loaded,
        channels=3,
        num_samples=num_samples
    )

    samples = denorm(samples)
    path = os.path.join(TRAIN_DATA_PATH, f"generated_samples.png")
    utils.save_image(samples, path, nrow=4)
    print(f"saved generated samples: {path}")

if __name__ == "__main__":
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
    )

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    g = torch.Generator()
    g.manual_seed(SEED)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CURRENT DEVICE:", DEVICE)


    # num_workers > 0일경우, windows 상에서 sub프로세스 하나 새로 띄움. 멀티프로세싱 꼬여서 RuntimeERR 발생가능
    trainset = torchvision.datasets.CIFAR10(root='./CNN_MyModel/MyModel/dataPT/',train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=TRAIN_BATCH_SIZE,
                                            shuffle=True, num_workers=0,
                                            generator=g)

    testset = torchvision.datasets.CIFAR10(root='./CNN_MyModel/MyModel/dataPT/',train=False,
                                            download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=TEST_BATCH_SIZE,
                                            shuffle=False, num_workers=0)
    
    #TrainDiffusion()
    #LoadModelAndGenerate()
