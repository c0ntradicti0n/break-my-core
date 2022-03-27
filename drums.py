import glob
from collections import defaultdict
import random
from pprint import pprint

import numpy

samples = list(glob.glob("./data/*.WAV"))
compass = len(samples) - 1
resolution = 0.125
theme_len = 16
voices = 10
piece_len = 14 * 14
beats_per_bar = 3
tempo = 200
sections_in_piece = 100
fs = 44100

import sounddevice as sd
import numpy as np
import wave

from typing import Tuple
from pathlib import Path


# Utility function that reads the whole `wav` file content into a numpy array
def wave_read(filename: Path) -> Tuple[np.ndarray, int]:
    with wave.open(str(filename), 'rb') as f:
        buffer = f.readframes(f.getnframes())
        inter = np.frombuffer(buffer, dtype=f'int{f.getsampwidth() * 8}')
        return np.reshape(inter, (-1, f.getnchannels())), f.getframerate()


def time_transpose(l, t):
    return [t() + e for e in l]


def time_add(ll, pos=0):
    rr = []
    for l in ll:
        r = []
        for (tone_height, tone_length) in l:
            r.append(
                (tone_height, tone_length + pos)
            )
            pos += tone_length
        rr.append(r)
    return rr


start = None


def generate_tone():
    """
    height, length

    :return:
    """
    return (random.randint(0, compass), int(((random.random() * beats_per_bar) + 1) / resolution ) * resolution)


def compose():
    global start
    melodies = defaultdict(list)
    theme = [generate_tone() for _ in range(theme_len)]

    for voice in range(voices):
        start = voice * beats_per_bar * 2

        melodies[voice] = [theme] + [
            theme
            if random.randint(0, voices) > 0.2 * voices else
            [generate_tone()
             for _ in
             range(theme_len)
             ]
            for _ in range(sections_in_piece)
        ]

    return melodies


melodies = compose()

pprint(melodies)

sounds = {i:
              wave_read(wav_file) for i, wav_file in enumerate(samples)
          }


def stretch_sample(sample, l):
    l_sample = len(sample) / fs

    rest = int(fs * (resolution / l - l_sample) * 60 / tempo)
    print(rest)
    if rest > 0:
        samples = np.vstack([sample] + [np.zeros((rest, 1))])
    if rest <= 0:
        samples = sample[:rest]
    return samples


def play(voice, melody):
    for section in melody:
        print(section)

        track = numpy.vstack(
            [stretch_sample(sounds[tone_height][0], tone_length) for tone_height, tone_length in section])

        print(f"new track in {voice}")
        sd.play(track, samplerate=fs, blocking=True)


# play(0, melodies[0])

from threading import Thread

for voice, melody in melodies.items():
    Thread(target=play, args=(voice, melody)).start()
