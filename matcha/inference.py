# import sys
# sys.path.insert(0, '/share/nas169/wago/Matcha-TTS')

import datetime as dt
from pathlib import Path

import IPython.display as ipd
import numpy as np
import soundfile as sf
import torch
from tqdm.auto import tqdm

# Hifigan imports
from matcha.hifigan.config import v1
from matcha.hifigan.denoiser import Denoiser
from matcha.hifigan.env import AttrDict
from matcha.hifigan.models import Generator as HiFiGAN
# Matcha imports
from matcha.models.matcha_tts import MatchaTTS
from matcha.text import sequence_to_text, text_to_sequence
from matcha.utils.model import denormalize
from matcha.utils.utils import get_user_data_dir, intersperse

def load_model(checkpoint_path, device):
    model = MatchaTTS.load_from_checkpoint(checkpoint_path, map_location=device)
    model.eval()
    return model

def load_vocoder(checkpoint_path):
    h = AttrDict(v1)
    hifigan = HiFiGAN(h).to(device)
    hifigan.load_state_dict(torch.load(checkpoint_path, map_location=device)['generator'])
    _ = hifigan.eval()
    hifigan.remove_weight_norm()
    return hifigan

@torch.inference_mode()
def process_text(text: str):
    x = torch.tensor(intersperse(text_to_sequence(text, ['parse_ipa'])[0], 0),dtype=torch.long, device=device)[None]
    x_lengths = torch.tensor([x.shape[-1]],dtype=torch.long, device=device)
    x_phones = sequence_to_text(x.squeeze(0).tolist())
    return {
        'x_orig': text,
        'x': x,
        'x_lengths': x_lengths,
        'x_phones': x_phones
    }


@torch.inference_mode()
def synthesise(text, spks=None):
    text_processed = process_text(text)
    start_t = dt.datetime.now()
    output = model.synthesise(
        text_processed['x'], 
        text_processed['x_lengths'],
        n_timesteps=n_timesteps,
        temperature=temperature,
        spks=spks,
        length_scale=length_scale
    )
    # merge everything to one dict    
    output.update({'start_t': start_t, **text_processed})
    return output

@torch.inference_mode()
def to_waveform(mel, vocoder):
    audio = vocoder(mel).clamp(-1, 1)
    audio = denoiser(audio.squeeze(0), strength=0.00025).cpu().squeeze()
    return audio.cpu().squeeze()
    
def save_to_folder(filename: str, output: dict, folder: str):
    folder = Path(folder)
    folder.mkdir(exist_ok=True, parents=True)
    np.save(folder / f'{filename}', output['mel'].cpu().numpy())
    sf.write(folder / f'{filename}.wav', output['waveform'], 22050, 'PCM_24')

if __name__=='__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
    
    MATCHA_CHECKPOINT = "/mnt/md1/user_wago/Matcha-TTS/logs/train/matbn_weight/runs/2024-10-24_06-06-23/checkpoints/last.ckpt"
    HIFIGAN_CHECKPOINT = "/mnt/md1/user_wago/hifigan/LJ_V1/generator_v1"
    OUTPUT_FOLDER = "synth_output"
    
    count_params = lambda x: f"{sum(p.numel() for p in x.parameters()):,}"

    model = load_model(MATCHA_CHECKPOINT, device)
    print(f"Model loaded! Parameter count: {count_params(model)}")
    
    vocoder = load_vocoder(HIFIGAN_CHECKPOINT)
    denoiser = Denoiser(vocoder, mode='zeros')

    texts = [
        "ɕ-i_55-t-ien_214-t͡ɕ-yn_55-ɕ-iau_51 n-ien_35-t-u_51 ʈ͡ʂ-au_55-ʂ-ɔu_55 tʰ-ai_35-uan_55 ɕ-ye_35-ʂ-əŋ_55 ʈ͡ʂ-ən_55-ʂ-ʅ_51 t͡ɕ-in_55-tʰ-ien_55 t͡ɕ-y_214-ɕ-iŋ_35"
    ]
    ## Number of ODE Solver steps
    n_timesteps = 10

    ## Changes to the speaking rate
    length_scale=1.0

    ## Sampling temperature
    temperature = 0.667
    
    outputs, rtfs = [], []
    rtfs_w = []
    for i, text in enumerate(tqdm(texts)):
        output = synthesise(text) #, torch.tensor([15], device=device, dtype=torch.long).unsqueeze(0))
        output['waveform'] = to_waveform(output['mel'], vocoder)

        # Compute Real Time Factor (RTF) with HiFi-GAN
        t = (dt.datetime.now() - output['start_t']).total_seconds()
        rtf_w = t * 22050 / (output['waveform'].shape[-1])

        ## Pretty print
        print(f"{'*' * 53}")
        print(f"Input text - {i}")
        print(f"{'-' * 53}")
        print(output['x_orig'])
        print(f"{'*' * 53}")
        print(f"Phonetised text - {i}")
        print(f"{'-' * 53}")
        print(output['x_phones'])
        print(f"{'*' * 53}")
        print(f"RTF:\t\t{output['rtf']:.6f}")
        print(f"RTF Waveform:\t{rtf_w:.6f}")
        rtfs.append(output['rtf'])
        rtfs_w.append(rtf_w)

        ## Display the synthesised waveform
        ipd.display(ipd.Audio(output['waveform'], rate=22050))

        ## Save the generated waveform
        save_to_folder(i, output, OUTPUT_FOLDER)

    print(f"Number of ODE steps: {n_timesteps}")
    print(f"Mean RTF:\t\t\t\t{np.mean(rtfs):.6f} ± {np.std(rtfs):.6f}")
    print(f"Mean RTF Waveform (incl. vocoder):\t{np.mean(rtfs_w):.6f} ± {np.std(rtfs_w):.6f}")