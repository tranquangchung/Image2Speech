from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import soundfile as sf
import torch


def map_to_array(batch):
  speech, _ = sf.read(batch["file"])
  batch["speech"] = speech
  return batch

def init_model():
  #### V1 ####
  processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")
  model = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")
  #### V2 ####
  # processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
  # model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
  #### V3 ####
  # processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-conformer-rel-pos-large-960h-ft")
  # model = Wav2Vec2ConformerForCTC.from_pretrained("facebook/wav2vec2-conformer-rel-pos-large-960h-ft")
  #### V4 ####
  # processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
  # model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")

  model.to("cuda")
  return processor, model

def init_model_vn():
    processor = Wav2Vec2Processor.from_pretrained("nguyenvulebinh/wav2vec2-base-vietnamese-250h")
    model = Wav2Vec2ForCTC.from_pretrained("nguyenvulebinh/wav2vec2-base-vietnamese-250h")
    model.to("cuda")
    return processor, model

def asr_english(path, model, processor):
  ds = map_to_array({"file": path})
  inputs = processor(ds["speech"], sampling_rate=16_000, return_tensors="pt", padding=True)
  with torch.no_grad():
    logits = model(inputs.input_values.to("cuda"), attention_mask=inputs.attention_mask.to("cuda")).logits
    # logits = model(inputs.input_values.to("cuda")).logits
  predicted_ids = torch.argmax(logits, dim=-1)
  transcription = processor.batch_decode(predicted_ids)
  return transcription

def asr_vietnamese(path, model, processor):
  ds = map_to_array({"file": path})
  inputs = processor(ds["speech"], sampling_rate=16_000, return_tensors="pt", padding=True)
  with torch.no_grad():
    logits = model(inputs.input_values.to("cuda")).logits
  predicted_ids = torch.argmax(logits, dim=-1)
  transcription = processor.batch_decode(predicted_ids)
  return transcription