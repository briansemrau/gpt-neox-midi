import argparse
import mido
import os
import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)
import tools.preprocess_data
from threading import Semaphore
from midi.tokenizer_tools import decode_str_to_bytes
import io
import tarfile
import multiprocessing
import copy
import uuid
import functools
import time
import tqdm
from typing import List


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to MIDI archive to augment.",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="",
        help="Path to output augmented MIDI archive.",
    )
    parser.add_argument(
        "--transpose", "-t",
        type=str,
        default="1,2,3,4,5,6,-1,-2,-3,-4,-5",
        help="List of comma-separated semitones to transpose by.",
    )
    parser.add_argument(
        "--time-stretch", "-s",
        type=str,
        default="0.025,0.05,0.1,-0.025,-0.05,-0.1",
        help="List of comma-separated uniform time stretch factors.",
    )
    parser.add_argument(
        "--workers", "-n",
        type=int,
        default=1,
        help="Number of workers to use for parallel processing.",
    )
    args = parser.parse_args()
    if args.output == "":
        input_root, input_ext = os.path.splitext(args.input)
        args.output = input_root + "_augmented"
        print(f"Output path not specified, using {args.output}.")
    output_ext = os.path.splitext(args.output)[1]
    if output_ext != ".tar.gz" and output_ext != "":
        print(f"Output archive must be a tar.gz file. ")
        sys.exit(1)
    if output_ext == "":
        args.output += ".tar.gz"
    if os.path.exists(args.output):
        print(f"Output path {args.output} already exists.")
        sys.exit(1)
    if not os.path.exists(os.path.dirname(args.output)):
        os.makedirs(os.path.dirname(args.output))
    
    return args


def augment_midi(bytes: bytes, transpose: List[int], time_stretch: List[float]) -> List[bytes]:
    file = io.BytesIO(bytes)
    try:
        midi_file = mido.MidiFile(file=file)
    except:
        return []
    augmented = []
    for xpose in transpose:
        for stretch in time_stretch:
            file.seek(0)
            aug_file = copy.deepcopy(midi_file)
            for track in aug_file.tracks:
                for message in track:
                    if message.type in ('note_on','note_off'):
                        message.note += xpose
                    #if hasattr(message, 'time'):
                    message.time = int(message.time * (1 + stretch))
            bytes_file = io.BytesIO()
            aug_file.save(file=bytes_file)
            augmented.append(bytes_file.getvalue())
    
    return augmented


def augment_midi_encoded(text: str, transpose: List[int], time_stretch: List[float]) -> List[bytes]:
    bytes = decode_str_to_bytes(text)
    return augment_midi(bytes, transpose, time_stretch)


def main():
    args = get_args()

    transpose = [int(x) for x in args.transpose.split(",")] if args.transpose else [0]
    time_stretch = [float(x) for x in args.time_stretch.split(",")] if args.time_stretch else [0.0]

    semaphore = Semaphore(1000)
    files = tools.preprocess_data.yield_from_files([args.input], semaphore)

    if args.workers > 1:
        pool = multiprocessing.Pool(processes=args.workers)
        augmented_docs = pool.imap(
            functools.partial(augment_midi_encoded, transpose=transpose, time_stretch=time_stretch),
            files,
            chunksize=32
        )
    else:
        augmented_docs = map(
            functools.partial(augment_midi_encoded, transpose=transpose, time_stretch=time_stretch),
            files
        )

    proc_start = time.time()
    total_bytes_processed = 0
    pbar = tqdm.tqdm()
    with tarfile.open(args.output, "w:gz") as tar:
        for i, aug_docs in enumerate(augmented_docs, start=1):
            semaphore.release()

            base_filename = str(uuid.uuid4())
            for n, aug_doc in enumerate(aug_docs):
                filename = f"{base_filename}_{n}.mid"
                tarinfo = tarfile.TarInfo(name=filename)
                tarinfo.size = len(aug_doc)
                total_bytes_processed += tarinfo.size
                tar.addfile(tarinfo, fileobj=io.BytesIO(aug_doc))

            # log progress
            if i % 1 == 0:
                current = time.time()
                elapsed = current - proc_start
                mbs = total_bytes_processed / elapsed / 1024 / 1024
                pbar.set_description(
                    f"Processed {i} documents ({i / elapsed} docs/s, {mbs} MB/s)."
                )
                if i != 0:
                    pbar.update(1)


if __name__ == "__main__":
    main()
