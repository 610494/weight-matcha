<div align="center">

# breezyvoice Matcha-TTS

This repo is from [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS)
We use [breezyvoice](https://arxiv.org/abs/2501.17790) to generate data and train Matcha-TTS on them.

## Inference

1. Set Up Path
To use local symbols, modify line 18 in inference.py to point to the path of this repository.

2. Set Checkpoint Path
Modify line 93 in inference.py to the TTS checkpoint you want to use.

3. Download HiFi-GAN Checkpoint
To generate waveforms, you need to download the vocoder from:
https://drive.google.com/drive/folders/1-eEYTB5Av9jNql0WGBlRoi-WH2J7bp5Y
The default checkpoint is LJ_V1. If you want to use a different checkpoint, modify line 94 in inference.py.

4. Set Output Path
The default output path is: synth_output/<checkpoint>
If you want to change it, modify line 96 in inference.py.

5. Set Data Path
Modify line 99 in inference.py to the path of your input data.
The data should follow the format:
<audio_path>|<text>