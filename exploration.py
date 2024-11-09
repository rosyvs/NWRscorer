#%%
import os
from transformers import AutoProcessor, AutoModelForCTC, Wav2Vec2Processor
import torch
from itertools import groupby
import evaluate
from jiwer import wer as jiwer_wer
# from datasets import load_dataset
import torchaudio
# from torch.utils.data import Dataset, DataLoader, DatasetFolder

# %% read transcription xlsx and convert to text files per audio
import pandas as pd
tx_file = '/Users/roso8920/Emotive Computing Dropbox/Rosy Southwell/SLHS/nonword repetition IPA.xlsx'
tx_df = pd.read_excel(tx_file)
tx_df['audio_filename'] = tx_df['Item'].apply(lambda x: f'NonwordRepetition_Item_{x}.wav')

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
    prediction = "".join(phonemes)
    prediction.replace(processor.tokenizer.word_delimiter_token, " ")

    # whether to ignore IPA stress marks
    if ignore_stress == True:
        prediction = prediction.replace("ˈ", "").replace("ˌ", "")
    return prediction


#%%
checkpoint = "bookbot/wav2vec2-ljspeech-gruut"
eval_wer = evaluate.load('wer')
model = AutoModelForCTC.from_pretrained(checkpoint)
processor = AutoProcessor.from_pretrained(checkpoint)
sr = processor.feature_extractor.sampling_rate

#%%
# let's look at the tokenizer. New IDs have been added to give 43 tokens (similar to 44in english IPA, coincidence?)
processor.tokenizer.get_vocab()

#%%
res_df = []
test_dir = '/Users/roso8920/Emotive Computing Dropbox/Rosy Southwell/SLHS/Audio/'
for i, row in tx_df.iterrows():
    test_file = os.path.join(test_dir, row['audio_filename'])
    label = row['IPA Translation']
    audio_array, sr = torchaudio.load(test_file)

    inputs = processor(torch.squeeze(audio_array), return_tensors="pt", padding=True, sampling_rate=sr)

    with torch.no_grad():
        res = model(inputs.input_values)
    logits = res.logits
    predicted_ids = torch.argmax(logits, dim=-1)

    pred = processor.decode(predicted_ids[0], output_word_offsets=True)
    word_offsets_ix = pred["word_offsets"]
    # convert to time rela to audio: Word offsets can be used in combination with the sampling rate and model downsampling rate to compute the time-stamps of transcribed words.
    word_offsets_samples = word_offsets_ix * 49

    prediction = decode_phonemes(predicted_ids[0], processor, ignore_stress=False)

    # use WER to compute accuracy, treating each phoneme as a word
    wer = jiwer_wer(' '.join([p for p in label.strip("/")]), ' '.join([p for p in prediction]))
    true_phoneme_count = len(label.strip("/"))
    hyp_phoneme_count = len(prediction)
    print(f'target: {label.strip("/")}\t hyp: {prediction} \t PER: {wer:.2f} \t phoneme count: target={true_phoneme_count} hyp={hyp_phoneme_count}')
    res_df.append({'audio_filename': row['audio_filename'], 'target': label.strip("/"), 'hyp': prediction, 'PER': wer, 'true_phoneme_count': true_phoneme_count, 'hyp_phoneme_count': hyp_phoneme_count})
res_df = pd.DataFrame(res_df)
res_df.to_csv('nonword_repetition_ASR_results_from_wav2vec2-ljspeech-gruut.csv', index=False)
# %%
