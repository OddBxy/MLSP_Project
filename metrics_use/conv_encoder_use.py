import torch
import torch.nn as nn

import torchaudio
import torchaudio.functional as F


class conv_encoder(nn.Module):
    def __init__(self, K=32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=K, kernel_size=3, stride=2, padding=1)

        self.convt1 = nn.ConvTranspose2d(in_channels=K, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.convt2 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.convt3 = nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=3, stride=2, padding=1)

        self.nonl = nn.ReLU()

    def forward(self, x, verbose=False):
        h1 = self.conv1(x)
        h1 = self.nonl(h1)

        h2 = self.conv2(h1)
        h2 = self.nonl(h2)

        h3 = self.conv3(h2)

        h4 = self.convt1(h3)
        h4 = self.nonl(h4)

        h5 = self.convt2(h4)
        h5 = self.nonl(h5)

        xhat = self.convt3(h5)
        if verbose:
          print(h1.shape)
          print(h2.shape)
          print(h3.shape)

          print(h4.shape)
          print(h5.shape)
          print(xhat.shape)
        return xhat, h3









def train_model(musics_tr, model):
    model = model.to('cuda')
    model.train()
    opt = torch.optim.Adam(params=model.parameters(), lr=1e-4)

    batch_size = 4
    num_epochs = 2000
    N = musics_tr.shape[0]

    criterion = nn.MSELoss()

    for e in range(num_epochs):
        model.train()

        # shuffle indices
        perm = torch.randint(0, N, (batch_size,))

        #batch = musics_tr[e:e+batch_size].unsqueeze(1).to("cuda")
        batch = musics_tr[perm].unsqueeze(1).to("cuda")
        opt.zero_grad()

        out, _ = model.forward(batch)
        H, W = out.shape[-2], out.shape[-1]
        batch = batch[..., :H, :W]

        #loss = ((out - batch)**2).mean() #using MSE loss
        loss = criterion(out, batch)
        loss.backward()

        opt.step()

        if e % 100 == 0:
            print('ep {} loss val {}'.format(e, loss.item()))



def test_model(musics_test, model):
    model.eval()
    with torch.no_grad():
        batch = musics_test[:].unsqueeze(1).to("cuda")
        out, _ = model(batch)

        # crop batch to match output
        H, W = out.shape[-2], out.shape[-1]
        batch_cropped = batch[..., :H, :W]

        loss = nn.MSELoss()(out, batch_cropped)
        print("Test batch MSE:", loss.item())




