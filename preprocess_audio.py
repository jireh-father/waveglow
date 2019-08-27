# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************\
import os
import random
import argparse
import json
import torch
import torch.utils.data
import sys
import numpy as np
import librosa
from multiprocess.pool import Pool

# We're using the audio processing from TacoTron2 to make sure it matches
sys.path.insert(0, 'tacotron2')
from tacotron2.layers import TacotronSTFT

MAX_WAV_VALUE = 32768.0


def files_to_list(filename):
    """
    Takes a text file of filenames and makes a list of filenames
    """
    with open(filename, encoding='utf-8') as f:
        files = f.readlines()

    files = [f.rstrip() for f in files]
    return files


def load_wav_to_torch(full_path, sampling_rate):
    data = librosa.core.load(full_path, sr=sampling_rate)[0]
    data = data / np.abs(data).max() * 0.999
    return torch.FloatTensor(data.astype(np.float32))


# def load_wav_to_torch(full_path):
#     """
#     Loads wavdata into torch array
#     """
#     sampling_rate, data = read(full_path)
#     return torch.from_numpy(data).float(), sampling_rate

int16_max = (2 ** 15) - 1
audio_norm_target_dBFS = -30


def preprocess_wav(fpath_or_wav, source_sr=None, sampling_rate=16000):
    # Load the wav from disk if needed
    if isinstance(fpath_or_wav, str):
        wav, source_sr = librosa.load(fpath_or_wav, sr=None)
    else:
        wav = fpath_or_wav

    # Resample the wav if needed
    if source_sr is not None and source_sr != sampling_rate:
        wav = librosa.resample(wav, source_sr, sampling_rate)

    # Apply the preprocessing: normalize volume and shorten long silences
    wav = normalize_volume(wav, audio_norm_target_dBFS, increase_only=True)
    wav = wav / np.abs(wav).max() * 0.999
    # wav = trim_long_silences(wav)
    return wav


def normalize_volume(wav, target_dBFS, increase_only=False, decrease_only=False):
    if increase_only and decrease_only:
        raise ValueError("Both increase only and decrease only are set")
    rms = np.sqrt(np.mean((wav * int16_max) ** 2))
    wave_dBFS = 20 * np.log10(rms / int16_max)
    dBFS_change = target_dBFS - wave_dBFS
    if dBFS_change < 0 and increase_only or dBFS_change > 0 and decrease_only:
        return wav
    return wav * (10 ** (dBFS_change / 20))


class Mel2Samp(torch.utils.data.Dataset):
    """
    This is the main class that calculates the spectrogram and returns the
    spectrogram, audio pair.
    """

    def __init__(self, training_files, segment_length, filter_length,
                 hop_length, win_length, sampling_rate, mel_fmin, mel_fmax, num_workers):
        self.audio_files = files_to_list(training_files)

        random.seed(1234)
        random.shuffle(self.audio_files)
        self.stft = TacotronSTFT(filter_length=filter_length,
                                 hop_length=hop_length,
                                 win_length=win_length,
                                 sampling_rate=sampling_rate,
                                 mel_fmin=mel_fmin, mel_fmax=mel_fmax)
        self.segment_length = segment_length
        self.sampling_rate = sampling_rate
        self.num_workers = num_workers

    def get_mel(self, audio):
        # audio_norm = audio / MAX_WAV_VALUE
        audio_norm = audio.unsqueeze(0)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        melspec = self.stft.mel_spectrogram(audio_norm)
        melspec = torch.squeeze(melspec, 0)
        return melspec

    def __getitem__(self, index):
        # Read audio
        filename = self.audio_files[index]
        audio_filename = self.training_audio_files[index]

        mel = torch.load(filename)
        # mel = torch.autograd.Variable(mel.cuda())
        # mel = torch.unsqueeze(mel, 0)
        # mel = mel.half() if self.is_fp16 else mel

        audio = load_wav_to_torch(audio_filename, self.sampling_rate)
        # if sampling_rate != self.sampling_rate:
        #     raise ValueError("{} SR doesn't match target {} SR".format(
        #         sampling_rate, self.sampling_rate))

        # Take segment
        if audio.size(0) >= self.segment_length:
            max_audio_start = audio.size(0) - self.segment_length
            audio_start = random.randint(0, max_audio_start)
            audio = audio[audio_start:audio_start + self.segment_length]
        else:
            audio = torch.nn.functional.pad(audio, (0, self.segment_length - audio.size(0)), 'constant').data
        mel = self.get_mel(audio)
        # audio = audio / MAX_WAV_VALUE

        return (mel, audio)

    def __len__(self):
        return len(self.audio_files)


# ===================================================================
# Takes directory of clean audio and makes directory of spectrograms
# Useful for making test sets
# ===================================================================
if __name__ == "__main__":
    # Get defaults so it can work with no Sacred
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--filelist_path", required=True)
    parser.add_argument('-c', '--config', type=str,
                        help='JSON file for configuration')
    parser.add_argument('-o', '--output_dir', type=str,
                        help='Output directory')
    parser.add_argument('-s', '--sampling_rate', type=int,
                        help='sample rate', default=22050)
    parser.add_argument('-n', '--num_processes', type=int,
                        help='num_processes', default=4)
    args = parser.parse_args()

    with open(args.config) as f:
        data = f.read()
    data_config = json.loads(data)["data_config"]
    train_config = json.loads(data)["train_config"]
    mel2samp = Mel2Samp(**data_config)


    def local_mel2samp(filepath):
        print("start", filepath)
        filepath = filepath.split("|")[0]
        audio = preprocess_wav(filepath, sampling_rate=args.sampling_rate)
        filename = os.path.basename(filepath)
        new_filepath = args.output_dir + '/' + filename + '.npy'
        print("finish", new_filepath)
        np.save(new_filepath, audio)

    filepaths = files_to_list(args.filelist_path)

    # Make directory if it doesn't exist
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
        os.chmod(args.output_dir, 0o775)

    with Pool(args.num_processes) as pool:  # ThreadPool(8) as pool:
        # list(tqdm(pool.imap(preprocess_speaker, speaker_dirs), dataset_name, len(speaker_dirs),
        list(pool.map(local_mel2samp, filepaths))

    # for filepath in filepaths:
    #     filepath = filepath.split("|")[0]
    #     audio = preprocess_wav(filepath, sampling_rate=args.sampling_rate)
    #     filename = os.path.basename(filepath)
    #     new_filepath = args.output_dir + '/' + filename + '.npy'
    #     print(new_filepath)
    #     np.save(new_filepath, audio)