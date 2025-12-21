import torch
import torchaudio
import os
from metrics_use.conv_encoder_use import conv_encoder, train_model, test_model
from metrics_use.metrics import compute_FAD, compute_mmd



def split_wav(wav, num_splits=2):
    c, L = wav.shape
    split_len = L // num_splits
    splits = []
    for i in range(num_splits):
        start = i * split_len
        end = (i + 1) * split_len if i < num_splits - 1 else L
        splits.append(wav[:, start:end])
    return splits


def get_spectrogram(wav):
  # convert to mono
  wav = wav.mean(dim=0, keepdim=True)

  #get spectrogram
  wav_stft = torch.stft(wav, 1024, window=torch.hamming_window(1024), return_complex=True, hop_length=256).abs()
  wav_stft = torch.log1p(wav_stft)
  wav_stft = torch.clamp(wav_stft, min=1e-6) #avoid 0 values

  # normalize
  wav_stft = (wav_stft - wav_stft.min()) / (wav_stft.max() - wav_stft.min() + 1e-6)

  return wav_stft




if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"


    # get all music
    path = 'data/slices/'
    files = os.listdir(path)

    stfts = []
    max_len = 0
    for i, fl in enumerate(files):
        wav, fs = torchaudio.load(path + fl)

        #split in 2 to avoid large vectors
        for wav_part in split_wav(wav, num_splits=2):

            wav_stft = get_spectrogram(wav_part)

            stfts.append(wav_stft.squeeze())
            if wav_stft.shape[-1] > max_len:
                max_len = wav_stft.shape[-1]


    musics = torch.stack(stfts)
    print(musics.shape)

    cutpt = int(musics.shape[0]*0.9)
    musics_tr = musics[:cutpt, :]
    musics_test = musics[cutpt:, :]
    print(musics_tr.shape)
    print(musics_test.shape)




    model = conv_encoder(K=32).to(device)
    model.load_state_dict(torch.load("metrics_use/model_weights/conv_encoder.pth",  map_location=torch.device(device)))
    model.eval()

    # create 2 audio groups
    music_group1 = musics_tr[:10].unsqueeze(0).transpose(0, 1).to(device)
    music_group2 = musics_test[:10].unsqueeze(0).transpose(0, 1).to(device)

    # Get models latents
    latents1 = None
    latents2= None
    with torch.no_grad():
        _, latents1 = model(music_group1)
        _, latents2 = model(music_group2)

    print(latents1.shape)
    print(latents2.shape)

    # moyenne en embeddings pour avoir "moins d'info", donc les comparaisons sont "moins sensibles" a de petit details
    embeddings1 = latents1.mean(dim=[2,3])  # shape [B, C]
    embeddings2 = latents2.mean(dim=[2,3])

    print(embeddings1.shape)
    print(embeddings2.shape)

    print("FAD score embedding moyenne : ", compute_FAD(embeddings1, embeddings2).item())
    print("MMD score embedding moyenne : ", compute_mmd(embeddings1, embeddings2).item())






