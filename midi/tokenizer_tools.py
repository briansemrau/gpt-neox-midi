"""
Copyright 2023 Brian Semrau.
Licensed under the MIT License.
"""

import numpy as np

def encode_bytes_to_utf8_str(data_bytes: bytes) -> str:
    # To maximize code reuse in the GPT NeoX codebase, we represent MIDI as UTF-8
    # To produce legal UTF-8, we encode into the Cyrillic block (arbitrarily)

    # UTF-8 requires the most significant bit, so we only have 7 bits to work with.
    # So we split our data into two bytes, using only 4 bits per byte.

    # original impl:
    #split_bytes = [((byte >> 4*(i%2)) & 0b1111)|0x80 for i, byte in enumerate([x for x in data_bytes for _ in range(2)])]
    #encoded_utf8_bytes = bytes([item for pair in zip([0xd0]*len(split_bytes), split_bytes) for item in pair])
    #encoded_str = encoded_utf8_bytes.decode("utf-8")
    #return encoded_str
    
    data_array = np.frombuffer(data_bytes, dtype=np.uint8)
    shift = np.tile(np.array([0, 4], dtype=np.uint8), data_array.shape[0])
    split_array = (((data_array.repeat(2) >> shift) & 0b1111) | 0x80).astype(np.uint8)
    encoded_array = np.empty(split_array.size * 2, dtype=np.uint8)
    encoded_array[::2] = 0xd0
    encoded_array[1::2] = split_array.reshape(-1)
    encoded_str = encoded_array.tobytes().decode("utf-8")

    return encoded_str

def decode_str_to_bytes(data_str: str) -> bytes:
    # original impl:
    #decoded_utf8_bytes = data_str.encode("utf-8")[1::2]
    #combined_bytes = [((b&0b1111)<<4)|(a&0b1111) for a, b in zip(decoded_utf8_bytes[::2], decoded_utf8_bytes[1::2])]
    #decoded_bytes = bytes(combined_bytes)
    #return decoded_bytes
    
    decoded_utf8_bytes = np.fromstring(data_str.encode("utf-8"), dtype=np.uint8)[1::2]
    pairs = decoded_utf8_bytes.reshape((-1, 2))
    combined_bytes = np.bitwise_or(np.bitwise_and(pairs[:, 1], 0b00001111) << 4, np.bitwise_and(pairs[:, 0], 0b00001111))
    decoded_bytes = combined_bytes.tobytes()

    return decoded_bytes
