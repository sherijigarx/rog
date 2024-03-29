import numpy as np
import torch
import torchaudio
from encodec import EncodecModel
from encodec.utils import convert_audio
from bhq.hubert_manager import HuBERTManager
from bhq.pre_kmeans_hubert import CustomHubert
from bhq.custumtokenizer import CustomTokenizer

large_quant_model = False  # Use the larger pretrained model
device = 'cuda'  # 'cuda', 'cpu', 'cuda:0', 0, -1, torch.device('cuda')

model = ('quantifier_V1_hubert_base_ls960_23.pth', 'tokenizer_large.pth') if large_quant_model else ('quantifier_hubert_base_ls960_14.pth', 'tokenizer.pth')

print('Loading HuBERT...')
hubert_model = CustomHubert(HuBERTManager.make_sure_hubert_installed(), device=device)
print('Loading Quantizer...')
quant_model = CustomTokenizer.load_from_checkpoint(HuBERTManager.make_sure_tokenizer_installed(model=model[0], local_file=model[1]), device)
print('Loading Encodec...')
encodec_model = EncodecModel.encodec_model_24khz()
encodec_model.set_target_bandwidth(6.0)
encodec_model.to(device)

print('Downloaded and loaded models!')


wav_file = 'speaker.wav'  # Put the path of the speaker you want to use here.
out_file = 'speaker.npz'  # Put the path to save the cloned speaker to here.

wav, sr = torchaudio.load(wav_file)

wav_hubert = wav.to(device)

if wav_hubert.shape[0] == 2:  # Stereo to mono if needed
    wav_hubert = wav_hubert.mean(0, keepdim=True)

print('Extracting semantics...')
semantic_vectors = hubert_model.forward(wav_hubert, input_sample_hz=sr)
print('Tokenizing semantics...')
semantic_tokens = quant_model.get_token(semantic_vectors)
print('Creating coarse and fine prompts...')
wav = convert_audio(wav, sr, encodec_model.sample_rate, 1).unsqueeze(0)

wav = wav.to(device)

with torch.no_grad():
    encoded_frames = encodec_model.encode(wav)
codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1).squeeze()

codes = codes.cpu()
semantic_tokens = semantic_tokens.cpu()

np.savez(out_file,
         semantic_prompt=semantic_tokens,
         fine_prompt=codes,
         coarse_prompt=codes[:2, :]
         )

print('Done!')