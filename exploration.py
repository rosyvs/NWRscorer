#%%
import os
from transformers import AutoProcessor, AutoModelForCTC, Wav2Vec2Processor
# import librosa
from scipy.io import wavfile
import torch
from itertools import groupby
from datasets import load_dataset
import torchaudio


#%% from https://huggingface.co/ct-vikramanantha/phoneme-scorer-v2-wav2vec2

def decode_phonemes(
    ids: torch.Tensor, processor: Wav2Vec2Processor, ignore_stress: bool = False
) -> str:
    """CTC-like decoding. First removes consecutive duplicates, then removes special tokens."""
    # removes consecutive duplicates
    ids = [id_ for id_, _ in groupby(ids)]

    special_token_ids = processor.tokenizer.all_special_ids + [
        processor.tokenizer.word_delimiter_token_id
    ]
    # converts id to token, skipping special tokens
    phonemes = [processor.decode(id_) for id_ in ids if id_ not in special_token_ids]

    # joins phonemes
    prediction = " ".join(phonemes)

    # whether to ignore IPA stress marks
    if ignore_stress == True:
        prediction = prediction.replace("ˈ", "").replace("ˌ", "")

    return prediction

checkpoint = "bookbot/wav2vec2-ljspeech-gruut"

model = AutoModelForCTC.from_pretrained(checkpoint)
processor = AutoProcessor.from_pretrained(checkpoint)
sr = processor.feature_extractor.sampling_rate


#%%
# or, read a single audio file #TODO not sure why using librosa here
test_file = '/Users/roso8920/Emotive Computing Dropbox/Rosy Southwell/SLHS/Audio/NonwordRepetition_Item_30.wav'
audio_array, sr = torchaudio.load(test_file)

inputs = processor(torch.squeeze(audio_array), return_tensors="pt", padding=True, sampling_rate=sr)

with torch.no_grad():
    logits = model(inputs["input_values"]).logits

predicted_ids = torch.argmax(logits, dim=-1)
prediction = decode_phonemes(predicted_ids[0], processor, ignore_stress=True)
# => should give /tægoʊdoʊvɪmlæpdeɪdʒaɪluːpaɪti/	
# %% read transcription xlsx and convert to text files per audio
import pandas as pd
tx_file = '/Users/roso8920/Emotive Computing Dropbox/Rosy Southwell/SLHS/nonword repetition IPA.xlsx'
tx_df = pd.read_excel(tx_file)
# %%
