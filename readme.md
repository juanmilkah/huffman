# Huffman Coder

A simple command-line tool for compressing and decompressing files using Huffman coding in Rust.

Compress a file:
```
./huffman --compress input.txt output.huff
```

Decompress a file:
```
./huffman --decompress input.huff output.txt
```

Handles empty files and single-symbol cases. Note: This is a basic implementation; not optimized for large files.
