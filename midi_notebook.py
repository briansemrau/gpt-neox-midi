#%%
# Load model

import os
import torch
import mido
from transformers import GPTNeoXForCausalLM
from midi.tokenizer import midi_encode, midi_decode
import midi.vocabulary

import transformers
transformers.logging.set_verbosity_error()

model_name = "neox-oore-gmp_aug_4k-19M"
model_path = f"/mnt/e/models/music/{model_name}/hf_model"
model = GPTNeoXForCausalLM.from_pretrained(model_path, local_files_only=True).cuda()

from transformers import RwkvForCausalLM
#model_name = "rwkv"
#model_path = f"/home/brian/RWKV-LM/RWKV-v4neo/out/hf_model"
#model = RwkvForCausalLM.from_pretrained(model_path, local_files_only=True).cuda()

#%%
# Generate

cwd = os.getcwd()  # vscode settings: "jupyter.notebookFileRoot": "${workspaceFolder}"

prompt_filename = None
prompt_mido = None
input_ids = None

unconditional = True
if not unconditional:
    prompt_filename = "pianocat.mid"
    prompt_mido = mido.MidiFile(filename=os.path.join(cwd, prompt_filename))
    input_ids = midi_encode(mid=prompt_mido)[0].unsqueeze(0).to(model.device)
else:
    input_ids = torch.tensor([[midi.vocabulary.start_token]]).to(model.device)

# generate beyond model max length
max_length = int(1024 * 4)
tokens_left = max_length - input_ids.shape[1]
model_max_length = model.config.max_position_embeddings
if model is RwkvForCausalLM:
    model_max_length = 999999999
gen_chunk_size = 64
all_tokens = input_ids
while tokens_left > 0:
    space_left = model_max_length - all_tokens.shape[1]
    max_gen = max(space_left, gen_chunk_size)
    n_to_gen = min(tokens_left, max_gen)
    tokens_left -= n_to_gen
    next_gen_tokens = model.generate(
        all_tokens[:,-(model_max_length-n_to_gen):],

        do_sample=True,
        #temperature=0.7,
        
        #typical_p=0.1,
        top_p=0.9,
        #top_k=3,
        
        repetition_penalty=1.05,
        
        #penalty_alpha=0.6,
        #top_k=10,
        #do_sample=False,

        #num_beams=4,
        #num_beam_groups=1,
        
        max_new_tokens=n_to_gen,
        suppress_tokens=[midi.vocabulary.end_token]
    )
    all_tokens = torch.cat((all_tokens, next_gen_tokens[:, -n_to_gen:]), dim=1)
all_tokens = all_tokens.to("cpu")[0]

tempo = 512820
# how to get tempo...
#if prompt_mido is not None:
#    tempo = prompt_mido.ticks_per_beat

output_file = f"{model_name}-output.mid"
if prompt_filename is not None:
    output_file = os.path.splitext(prompt_filename)[0] + "_" + output_file
gen_mido = midi_decode(index_list=all_tokens, fname=output_file, tempo=tempo)
gen_mido.save(os.path.join(cwd, output_file))
