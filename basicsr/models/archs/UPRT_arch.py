import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

#UPRT
class UPRT(nn.Module):
    def __init__(self,
        dim=48,
        num_blocks = [1,2,2], 
        num_refinement_blocks = 2,
        heads = [1,2,4],
        ffn_expansion_factor = 2,
        splits=[4,2,1]):
        super(UPRT,self).__init__()

        self.conv = nn.Conv2d(1,dim,3,1,1)
        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,split=splits[0]) for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,split=splits[1]) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,split=splits[2]) for i in range(num_blocks[2])])
        
        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,split=splits[1]) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1 
        self.reduce_chan_level1 = nn.Conv2d(int(dim*2**1), dim, kernel_size=1)
        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,split=splits[0]) for i in range(num_blocks[0])])
        
        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,split=splits[0]) for i in range(num_refinement_blocks)])
        
        self.output = nn.Conv2d(dim,2,3,1,1)
    def forward(self,x):

        #IFFB
        index = (x.detach() > 0).float()
        x = torch.fft.rfft2(x,norm='ortho')
        x[:,:,:4,:4] = 0
        x = torch.fft.irfft2(x,norm='ortho')
        x = x*index
        
        #encoder-decoder
        inp_enc_level1 = self.conv(x)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
     
        latent = self.latent(inp_enc_level3) 

        inp_dec_level2 = self.up3_2(latent)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2) 

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        inp_dec_level1 = self.reduce_chan_level1(inp_dec_level1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        
        out_dec_level1 = self.refinement(out_dec_level1)

        #output
        out_dec_level1 = self.output(out_dec_level1)
        return out_dec_level1

#TFFB
class Frequency2D(nn.Module):
    def __init__(self,dim,expand_factor=4):
        super(Frequency2D,self).__init__()
        self.norm = LayerNorm(dim)
        self.conv = nn.Conv2d(dim,dim,1)
        self.filter = nn.Sequential(nn.Conv2d(2,2*expand_factor,3,1,1),nn.LeakyReLU(0.2),nn.Conv2d(2*expand_factor,1,3,1,1),nn.Sigmoid())
        self.out = nn.Conv2d(dim,dim,1)
    def forward(self,x0):
        b,c,h,w = x0.shape
        x = self.norm(x0)
        x = self.conv(x)
        x_fft = torch.fft.rfft2(x)
        x_fft = torch.fft.fftshift(x_fft)
        filter = torch.abs(x_fft)
        filter = torch.cat([torch.max(filter,1)[0].unsqueeze(1),torch.mean(filter,1).unsqueeze(1)],dim=1)
        filter = self.filter(filter)
        x_fft = torch.complex(x_fft.real * filter, x_fft.imag* filter)
        x_fft = torch.fft.ifftshift(x_fft)
        x =  torch.fft.irfft2(x_fft,s=(h,w))
        x = self.out(x) + x0
        return x

#UPRTB    
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor,split=4):
        super(TransformerBlock, self).__init__()
        self.filter1 = Frequency2D(dim,4)
        self.norm1 = LayerNorm(dim)
        self.attn1 = LineFrequency(dim, num_heads,split)
        self.norm2 = LayerNorm(dim)
        self.ffn2 = MLP(dim, ffn_expansion_factor)
        self.norm3 = LayerNorm(dim)
        self.attn3 = WindowAttention(dim,num_heads,8,False)
        self.norm4 = LayerNorm(dim)
        self.ffn4 = MLP(dim, ffn_expansion_factor)
    def forward(self, x):
        x = self.filter1(x)
        x = x + self.attn1(self.norm1(x))
        x = x + self.ffn2(self.norm2(x))
        x = x + self.attn3(self.norm3(x))
        x = x + self.ffn4(self.norm4(x))
        return x

#LFPB
class LineFrequency(nn.Module):
    def __init__(self, dim, num_heads,split):
        super(LineFrequency,self).__init__()
        self.split = split
        self.q = nn.Sequential(nn.Conv2d(dim,dim,1),nn.Conv2d(dim,dim,3,1,1,groups=dim))
        self.conva = nn.Conv2d(dim//2,dim//2,1,groups=num_heads,bias=False)
        self.convb = nn.Conv2d(dim//2,dim//2,1,groups=num_heads,bias=False)
        self.convc = nn.Conv2d(dim//2,dim//2,1,groups=num_heads,bias=False)
        self.convd = nn.Conv2d(dim//2,dim//2,1,groups=num_heads,bias=False)
        self.project_out = nn.Conv2d(dim,dim,1)
    def forward(self,x):
        b,c,h,w = x.shape
        q1,q2 = self.q(x).chunk(2,dim=1)
        q1 = rearrange(q1, 'b c h (s w) -> (b s) c h w',s=self.split)
        q2 = rearrange(q2, 'b c (s h) w -> (b s) c h w',s=self.split)
        q1 = torch.fft.rfft(q1,dim=-1,norm='ortho')
        q2 = torch.fft.rfft(q2,dim=-2,norm='ortho')
        out1_real = self.conva(q1.real)-self.convb(q1.imag)
        out1_imag = self.convb(q1.real)+self.conva(q1.imag)
        out2_real = self.convc(q2.real)-self.convd(q2.imag)
        out2_imag = self.convd(q2.real)+self.convc(q2.imag)
        out1 = torch.fft.irfft(torch.complex(out1_real,out1_imag),dim=-1,norm='ortho')
        out2 = torch.fft.irfft(torch.complex(out2_real,out2_imag),dim=-2,norm='ortho')
        out1 = rearrange(out1, '(b s) c h w->b c h (s w)',s=self.split)
        out2 = rearrange(out2, '(b s) c h w->b c (s h) w',s=self.split)
        out = self.project_out(torch.cat([out1,out2],dim=1))
        return out

#WSA
class WindowAttention(nn.Module):
    def __init__(self,dim,heads=1,wsize = 8,shift =False):
        super().__init__()
        self.qkv = nn.Conv2d(dim,3*dim,1)
        self.wsize = wsize
        self.scale = (dim//heads) ** -0.5
        self.shift = shift
        self.softmax = nn.Softmax(dim=-1)
        self.proj = nn.Conv2d(dim,dim,1)
        self.heads = heads
    def forward(self,x):
        b,c,h,w = x.shape
        q,k,v = self.qkv(x).chunk(3,dim=1)
        if self.shift:
            q = torch.roll(q,shifts=(-self.wsize//2, -self.wsize//2), dims=(2,3))
            k = torch.roll(k,shifts=(-self.wsize//2, -self.wsize//2), dims=(2,3))
            v = torch.roll(v,shifts=(-self.wsize//2, -self.wsize//2), dims=(2,3))
        q = rearrange(q,'b (hed c) (h dh) (w dw)->(b h w) hed (dh dw) c',dh=self.wsize,dw =self.wsize,hed=self.heads)
        k = rearrange(k,'b (hed c) (h dh) (w dw)->(b h w) hed (dh dw) c',dh=self.wsize,dw =self.wsize,hed=self.heads)
        v = rearrange(v,'b (hed c) (h dh) (w dw)->(b h w) hed (dh dw) c',dh=self.wsize,dw =self.wsize,hed=self.heads)
        atn = torch.matmul(q,k.transpose(-1,-2)) * self.scale 
        atn = self.softmax(atn)
        y = torch.matmul(atn,v)
        y = rearrange(y,'(b h w) hed (dh dw) c->b (hed c) (h dh) (w dw)',h = h//self.wsize,w=w//self.wsize,dh=self.wsize,dw=self.wsize)
        if self.shift:
            y =  torch.roll(y, shifts=(self.wsize//2, self.wsize//2), dims=(2, 3))
        y = self.proj(y)
        return y

class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, 1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, 1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

#GDFN
class MLP(nn.Module):
    def __init__(self, dim,ratio=2):
        super(MLP,self).__init__()
        expandim = int(dim*ratio)
        self.proj1 = nn.Conv2d(dim,expandim,1)
        self.conv = nn.Conv2d(expandim,expandim,3,1,1,groups=expandim)
        self.projout = nn.Conv2d(dim,dim,1)
    def forward(self,x):
        x = self.proj1(x)
        x1,x2 = self.conv(x).chunk(2,dim=1)
        x = F.gelu(x1) * x2
        x = self.projout(x)
        return x

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
    def forward(self, x):
        h, w = x.shape[-2:]
        x = to_3d(x)
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        x = (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias
        return to_4d(x, h, w)
    
if __name__ == "__main__":
    net = UPRT()(dim=48)
    total = sum([param.nelement() for param in net.parameters()])
    print('   Number of params: %.2fM' % (total / 1e6))