
# Intro

I was able to do this on a desktop machine equipped with an AMD 7900X, 32GB RAM, and an RTX 3090.

# Setup GPT-NeoX

```sh
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -r requirements-midi.txt
pip install -r requirements-dev.txt
pip install -r requirements-flashattention.txt
pip install -r requirements-wandb.txt
pip install -r requirements-tensorboard.txt
```

# Download Lakh Midi dataset

https://colinraffel.com/projects/lmd/

```sh
DATA_DIR=/mnt/e/datasets/music/lakh_midi_v0.1

curl http://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz -o lmd_full.tar.gz
```

<!-- (optionally, but recommended) split into parts to avoid memory issues when tokenizing

```sh
PARTS=6

wget https://github.com/AQUAOSOTech/tarsplitter/releases/download/v2.2.0/tarsplitter_linux
mv tarsplitter_linux tarsplitter
chmod +x tarsplitter

gzip -d -c $DATA_DIR/lmd_full.tar.gz > $DATA_DIR/lmd_full.tar

mkdir $DATA_DIR/parts
./tarsplitter -m split -i $DATA_DIR/lmd_full.tar -o $DATA_DIR/parts/lmd_full-part -p $PARTS

for file in $DATA_DIR/parts/lmd_full-part*.tar; do gzip "$file" & done
```

if you don't want to split the dataset, it's easiest to just do this:

```sh
mkdir $DATA_DIR/parts
mv $DATA_DIR/lmd_full.tar.gz $DATA_DIR/parts/lmd_full-part0.tar.gz
``` -->

# Tokenize

I only modified GPT-NeoX to handle .tar.gz archives of midi files.
Modify [MidiReader](./tools/preprocess_data.py#L147) to add more file types.

<!-- ```sh
source venv/bin/activate

mkdir $DATA_DIR/tokenized
for file in $DATA_DIR/parts/lmd_full-part*.tar.gz; do python ./tools/preprocess_data.py --input $file --output-prefix "$DATA_DIR/tokenized/$(basename "$file" .tar.gz)" --dataset-impl mmap --workers $(nproc); done

python ./tools/merge_datasets --input $DATA_DIR/tokenized/ --output-prefix $DATA_DIR/tokenized/lmd_full
``` -->
```sh
source venv/bin/activate

mkdir $DATA_DIR/tokenized
python ./tools/preprocess_data.py --input $DATA_DIR/lmd_full.tar.gz --output-prefix $DATA_DIR/tokenized/lmd_full --dataset-impl mmap --workers $(nproc)
```

# train

Update [local_setup.yml](./local_setup.yml) to make sure the data paths are correct.

```sh
source venv/bin/activate
python ./deepy.py train.py -d configs midi_19M.yml local_setup.yml
```

convert to HF transformers

```sh
MODEL_DIR=/mnt/e/models/music/neox-oore-lmd-19M

python  ./tools/convert_v1.0_to_hf.py --input_dir $MODEL_DIR/$(cat $MODEL_DIR/latest) --config_file configs/midi_19M.yml --output_dir $MODEL_DIR/hf_model/
```

upload to HF

```sh
huggingface-cli login
python ./tools/upload.py
```

# inference

[midi_notebook.py](./midi_notebook.py)
