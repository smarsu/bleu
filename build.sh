set -e
set -x

g++ -std=c++11 -O3 -fPIC -shared bleu.cc -o libbleu.so
g++ -std=c++11 -O3 bleu.cc -o bleu -DWITH_MAIN

python3 bleu.py
