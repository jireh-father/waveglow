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
import mel2samp

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




# ===================================================================
# Takes directory of clean audio and makes directory of spectrograms
# Useful for making test sets
# ===================================================================
if __name__ == "__main__":
    # Get defaults so it can work with no Sacred
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--filelist_path", required=True)
    parser.add_argument('-o', '--output_dir', type=str,
                        help='Output directory')
    parser.add_argument('-s', '--sampling_rate', type=int,
                        help='sample rate', default=22050)
    parser.add_argument('-n', '--num_processes', type=int,
                        help='num_processes', default=4)
    args = parser.parse_args()


    def preprocess_wav(fpath_or_wav, sampling_rate=16000):
        # Load the wav from disk if needed
        if isinstance(fpath_or_wav, str):
            wav = librosa.core.load(fpath_or_wav, sr=sampling_rate)[0]
        else:
            wav = fpath_or_wav

        return wav

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
