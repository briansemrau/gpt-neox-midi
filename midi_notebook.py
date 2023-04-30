#%%
# Load model

import os
import torch
import mido
from transformers import GPTNeoXForCausalLM
from midi.tokenizer import midi_encode, midi_decode

model_dir = "/mnt/e/models/music/neox-oore-lmd-49M/hf_model"
model = GPTNeoXForCausalLM.from_pretrained(model_dir, local_files_only=True).cpu()


#%%
# Generate

cwd = os.getcwd()  # vscode settings: "jupyter.notebookFileRoot": "${workspaceFolder}"

prompt_filename = None
prompt_mido = None
input_ids = None

unconditional = True
if not unconditional:
    prompt_filename = "mm3-int.mid"
    prompt_mido = mido.MidiFile(filename=os.path.join(cwd, prompt_filename))
    input_ids = midi_encode(mid=prompt_mido)[0].unsqueeze(0).to(model.device)

gen_tokens = model.generate(
    input_ids,
    do_sample=True,

    temperature=0.9,
    
    #typical_p=0.2,
    
    penalty_alpha=0.6,
    top_k=4,
    
    max_length=512,
)[0]

if input_ids is not None:
    all_tokens = list(input_ids) + list(gen_tokens)
else:
    all_tokens = list(gen_tokens)

tempo = 512820
if prompt_mido is not None:
    tempo = prompt_mido.ticks_per_beat

output_file = "output.mid"
gen_mido = midi_decode(index_list=all_tokens, fname=output_file)
gen_mido.save(os.path.join(cwd, output_file))
