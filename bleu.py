# Copyright (c) 2020 smarsufan. All Rights Reserved.

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np
import ctypes
import time

libbleu = ctypes.cdll.LoadLibrary('libbleu.so')
libbleu.sentence_bleu.restype = ctypes.c_float


def ptrof(arr):
  return arr.ctypes.data_as(ctypes.c_void_p)


def unitest():
  while True:
    vocab_size = 15
    max_len = 20

    reference_len = int(np.random.randint(0, max_len, 1))
    candidate_len = int(np.random.randint(0, max_len, 1))

    reference = np.random.randint(0, vocab_size, reference_len).astype(np.int16)
    candidate = np.random.randint(0, vocab_size, candidate_len).astype(np.int16)

    t1 = time.time()
    score1 = sentence_bleu([reference], candidate, smoothing_function=SmoothingFunction().method1)
    t2 = time.time()
    score2 = libbleu.sentence_bleu(ptrof(reference), reference_len, ptrof(candidate), candidate_len)
    t3 = time.time()

    print('reference_len: {}, candidate_len: {}, \nreference: {}, \ncandidate: {}, \nscore1: {}, score2: {}, \nt1: {}, t2: {} \n'.format(
      reference_len, candidate_len, reference, candidate, score1, score2, t2 - t1, t3 - t2))

    if score1 - score2 > 1e-4:
      exit(1)


if __name__ == '__main__':
  unitest()

reference = [[1, 2, 3, 4, 5]]
candidate = [1, 2, 3, 4, 6]

score = sentence_bleu(reference, candidate, smoothing_function=SmoothingFunction().method1)
print(score)
