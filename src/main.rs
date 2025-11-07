use std::{
    collections::{BinaryHeap, HashMap},
    fs::File,
    io::{self, BufReader, BufWriter, Read, Write},
};

// Node represents a node in the Huffman tree.
// Leaf nodes have a character (ch) and frequency (freq).
// Internal nodes have children (left, right) and combined frequency.
#[derive(Debug, Eq, PartialEq)]
pub struct Node {
    pub ch: Option<u8>,
    pub freq: usize,
    pub left: Option<Box<Node>>,
    pub right: Option<Box<Node>>,
}

impl Node {
    // Creates a leaf node with a character and its frequency.
    pub fn new_leaf(ch: u8, f: usize) -> Self {
        Self {
            ch: Some(ch),
            freq: f,
            left: None,
            right: None,
        }
    }

    // Creates an internal node with combined frequency and child nodes.
    pub fn new_internal(freq: usize, left: Node, right: Node) -> Self {
        Self {
            ch: None,
            freq,
            left: Some(Box::new(left)),
            right: Some(Box::new(right)),
        }
    }
}

// HeapNode wraps Node for use in BinaryHeap (min-heap simulation via reverse ordering).
#[derive(Eq, PartialEq)]
pub struct HeapNode(Box<Node>);

// Implements Ord to create a min-heap based on frequency (lower freq first),
// with tie-breaking on character if present.
impl Ord for HeapNode {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other
            .0
            .freq
            .cmp(&self.0.freq)
            .then_with(|| self.0.ch.cmp(&other.0.ch))
    }
}

impl PartialOrd for HeapNode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

// Prints usage instructions for the program.
pub fn print_usage(program: &str) {
    eprintln!("Usage:");
    eprintln!("  {} --compress inputfile outputfile", program);
    eprintln!("  {} --decompress inputfile outputfile", program);
}

// Builds the Huffman tree from character frequencies using a priority queue.
pub fn build_tree(freqs: &HashMap<u8, usize>) -> Option<Node> {
    // Initialize min-heap with leaf nodes for each character.
    let mut heap = BinaryHeap::new();
    for (ch, f) in freqs.iter() {
        heap.push(HeapNode(Box::new(Node::new_leaf(*ch, *f))));
    }

    // Repeatedly merge the two lowest-frequency nodes until one remains.
    while heap.len() > 1 {
        let HeapNode(left) = heap.pop().unwrap();
        let HeapNode(right) = heap.pop().unwrap();

        // Create parent node with combined frequency.
        let parent = Node::new_internal(left.freq + right.freq, *left, *right);
        heap.push(HeapNode(Box::new(parent)));
    }

    heap.pop().map(|hn| *hn.0)
}

// Recursively traverses the Huffman tree to generate prefix codes and their lengths.
// - false for left branch (0), true for right branch (1).
pub fn generate_code_lengths(
    root: &Node,
    prefix: &mut Vec<bool>,
    codes: &mut HashMap<u8, Vec<bool>>,
    code_lengths: &mut HashMap<u8, u8>,
) {
    // Base case: leaf node - assign code and length.
    if root.left.is_none() && root.right.is_none() {
        if let Some(ch) = root.ch {
            codes.insert(
                ch,
                if prefix.is_empty() {
                    vec![false] // Special case for single-symbol tree: code '0'.
                } else {
                    prefix.clone()
                },
            );
            code_lengths.insert(ch, 1.max(prefix.len() as u8));
        }
        return;
    }

    // Traverse left child (append 0).
    if let Some(ref left) = root.left {
        prefix.push(false);
        generate_code_lengths(left, prefix, codes, code_lengths);
        prefix.pop();
    }

    // Traverse right child (append 1).
    if let Some(ref right) = root.right {
        prefix.push(true);
        generate_code_lengths(right, prefix, codes, code_lengths);
        prefix.pop();
    }
}

// Packs a sequence of bits (bools) into bytes for efficient storage.
pub fn pack_bits(bits: &[bool]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(bits.len().div_ceil(8));
    let mut byte = 0u8;
    let mut bit_count = 0;

    // Build bytes by shifting and setting bits.
    for bit in bits.iter() {
        byte <<= 1;
        if *bit {
            byte |= 1
        }
        bit_count += 1;
        if bit_count == 8 {
            bytes.push(byte);
            byte = 0;
            bit_count = 0;
        }
    }

    // Handle remaining bits in the last byte (pad with zeros on the right).
    if bit_count > 0 {
        byte <<= 8 - bit_count;
        bytes.push(byte);
    }
    bytes
}

// Writes the header to the output file: number of symbols, then symbol-length pairs.
pub fn write_header<W: Write>(writer: &mut W, code_lengths: &HashMap<u8, u8>) -> io::Result<()> {
    // Write symbol count as u32 LE.
    writer.write_all(&(code_lengths.len() as u32).to_le_bytes())?;
    // Write each symbol (u8) and its code length (u8).
    for (ch, l) in code_lengths {
        writer.write_all(&[*ch])?;
        writer.write_all(&[*l])?;
    }

    Ok(())
}

// Generates canonical Huffman codes from code lengths.
// Canonical codes are assigned in sorted order (by length, then symbol) for consistency.
pub fn generate_canonical_codes(code_lengths: &HashMap<u8, u8>) -> HashMap<u8, Vec<bool>> {
    // Create entries for sorting.
    let mut entries: Vec<CodeEntry> = code_lengths
        .iter()
        .map(|(&ch, &l)| CodeEntry {
            ch,
            length: l,
            code: 0,
        })
        .collect();

    // Sort by code length ascending, then by symbol value.
    entries.sort_by(|a, b| {
        if a.length == b.length {
            a.ch.cmp(&b.ch)
        } else {
            a.length.cmp(&b.length)
        }
    });

    // Assign codes sequentially, shifting for length changes.
    let mut code = 0u32;
    let mut prev_len = 0u8;
    for e in &mut entries {
        code <<= (e.length - prev_len) as u32;
        e.code = code;
        code += 1;
        prev_len = e.length;
    }

    // Convert numeric codes to bit vectors (MSB first).
    let mut codes = HashMap::new();
    for e in entries {
        let mut bits = Vec::with_capacity(e.length as usize);
        for i in 0..e.length {
            let bit = ((e.code >> (e.length - 1 - i)) & 1) == 1;
            bits.push(bit);
        }
        codes.insert(e.ch, bits);
    }
    codes
}

// Performs file compression: reads input, builds tree, encodes data, writes output.
pub fn compress_file(input: &str, output: &str) -> io::Result<()> {
    // Read entire input file into memory.
    let mut file = BufReader::new(File::open(input)?);
    let mut content = Vec::new();
    file.read_to_end(&mut content)?;

    // Handle empty file specially.
    if content.is_empty() {
        let mut output = BufWriter::new(File::create(output)?);
        output.write_all(&0u32.to_le_bytes())?; // zero symbols
        output.write_all(&0u32.to_le_bytes())?; // zero length
        output.write_all(&0u32.to_le_bytes())?; // zero bits
        return Ok(());
    }

    // Compute character frequencies.
    let mut freq: HashMap<u8, usize> = HashMap::new();
    for c in content.iter() {
        *freq.entry(*c).or_insert(0) += 1;
    }

    // Build Huffman tree (special case for single symbol).
    let root = if freq.len() == 1 {
        let (ch, f) = freq.iter().next().unwrap();
        Node::new_leaf(*ch, *f)
    } else {
        build_tree(&freq).unwrap()
    };

    // Generate tree-based codes and lengths (lengths used for header).
    let mut codes: HashMap<u8, Vec<bool>> = HashMap::new();
    let mut code_lengths: HashMap<u8, u8> = HashMap::new();
    generate_code_lengths(&root, &mut Vec::new(), &mut codes, &mut code_lengths);
    // Override with canonical codes for consistency with decompression.
    let codes = generate_canonical_codes(&code_lengths);

    // Encode content using canonical codes.
    let mut compressed_bits = Vec::new();
    for c in content.iter() {
        if let Some(c) = codes.get(c) {
            compressed_bits.extend_from_slice(c);
        }
    }

    // Pack bits into bytes.
    let packed_bits = pack_bits(&compressed_bits);

    // Write output: header, original length, bit length, packed data.
    let mut output = BufWriter::new(File::create(output)?);
    write_header(&mut output, &code_lengths)?;
    output.write_all(&(content.len() as u32).to_le_bytes())?;
    output.write_all(&(compressed_bits.len() as u32).to_le_bytes())?;
    output.write_all(&packed_bits)?;

    Ok(())
}

// Struct for code entries used in canonical code generation and tree rebuilding.
#[derive(Debug)]
pub struct CodeEntry {
    pub ch: u8,
    pub length: u8,
    pub code: u32,
}

// Rebuilds the Huffman tree from symbols and their code lengths using canonical ordering.
#[allow(clippy::collapsible_else_if)]
pub fn build_tree_from_codelengths(codes: &[u8], lengths: &[u8]) -> Node {
    // Create and sort entries (same as in generate_canonical_codes).
    let mut entries: Vec<CodeEntry> = codes
        .iter()
        .zip(lengths.iter())
        .map(|(&ch, &l)| CodeEntry {
            ch,
            length: l,
            code: 0,
        })
        .collect();

    entries.sort_by(|a, b| {
        if a.length == b.length {
            a.ch.cmp(&b.ch)
        } else {
            a.length.cmp(&b.length)
        }
    });

    // Assign canonical codes.
    let mut code = 0u32;
    let mut prev_len = 0u8;
    for e in &mut entries {
        code <<= (e.length - prev_len) as u32;
        e.code = code;
        code += 1;
        prev_len = e.length;
    }

    // Initialize root node.
    let mut root = Node {
        ch: None,
        freq: 0,
        left: None,
        right: None,
    };

    // Insert each symbol into the tree by following its code path (0 left, 1 right).
    for e in entries.iter() {
        let mut current = &mut root;
        for i in 0..e.length {
            let bit = (e.code >> (e.length - 1 - i)) & 1;
            if bit == 1 {
                if current.right.is_none() {
                    current.right = Some(Box::new(Node {
                        ch: None,
                        freq: 0,
                        left: None,
                        right: None,
                    }));
                }
                current = current.right.as_mut().unwrap();
            } else {
                if current.left.is_none() {
                    current.left = Some(Box::new(Node {
                        ch: None,
                        freq: 0,
                        left: None,
                        right: None,
                    }));
                }
                current = current.left.as_mut().unwrap();
            }
        }
        // Set leaf character.
        current.ch = Some(e.ch);
    }
    root
}

// Unpacks bytes back into a sequence of bits (up to original_bit_length).
pub fn unpack_bits(bytes: &[u8], original_bit_length: usize) -> Vec<bool> {
    let mut bits = Vec::with_capacity(original_bit_length);
    // Extract bits MSB first from each byte.
    for b in bytes.iter() {
        for i in (0..8).rev() {
            if bits.len() == original_bit_length {
                break;
            }
            bits.push(((b >> i) & 1) == 1);
        }
    }
    bits
}

// Reads the header from input: symbol count, then symbol-length pairs.
pub fn read_header<R: Read>(reader: &mut R) -> io::Result<(Vec<u8>, Vec<u8>)> {
    let mut count_bytes = [0u8; 4];
    reader.read_exact(&mut count_bytes)?;
    let symbol_count = u32::from_le_bytes(count_bytes) as usize;
    let mut symbols = Vec::with_capacity(symbol_count);
    let mut lengths = Vec::with_capacity(symbol_count);

    // Read each symbol and length.
    for _ in 0..symbol_count {
        let mut ch = [0u8; 1];
        reader.read_exact(&mut ch)?;
        symbols.push(ch[0]);

        let mut len = [0u8; 1];
        reader.read_exact(&mut len)?;
        lengths.push(len[0]);
    }

    Ok((symbols, lengths))
}

// Performs file decompression: reads input, rebuilds tree, decodes bits, writes output.
pub fn decompress_file(input: &str, output: &str) -> io::Result<()> {
    let mut input = BufReader::new(File::open(input)?);

    // Read header (symbols and lengths).
    let (symbols, lengths) = read_header(&mut input)?;
    // Read original file length (u32 LE).
    let mut original_length = [0u8; 4];
    input.read_exact(&mut original_length)?;
    let original_length = u32::from_le_bytes(original_length) as usize;

    // Read compressed bit length (u32 LE).
    let mut bit_length_bytes = [0u8; 4];
    input.read_exact(&mut bit_length_bytes)?;
    let bits_length = u32::from_le_bytes(bit_length_bytes) as usize;

    // Read packed compressed data.
    let mut packed_bits = Vec::new();
    input.read_to_end(&mut packed_bits)?;

    // Unpack to bits.
    let compressed_bits = unpack_bits(&packed_bits, bits_length);
    // Rebuild tree from lengths.
    let root = build_tree_from_codelengths(&symbols, &lengths);
    let mut decompressed = Vec::with_capacity(original_length);
    let mut current = &root;

    // Traverse tree bit by bit to decode symbols.
    for bit in compressed_bits {
        // Move left (0) or right (1).
        current = if !bit {
            current.left.as_ref().unwrap()
        } else {
            current.right.as_ref().unwrap()
        };

        // If leaf, output symbol and reset to root.
        if current.left.is_none() && current.right.is_none() {
            decompressed.push(current.ch.unwrap());
            current = &root;

            // Stop if reached original length.
            if decompressed.len() == original_length {
                break;
            }
        }
    }

    // Write decompressed data to output.
    let mut output = BufWriter::new(File::create(output)?);
    output.write_all(&decompressed)?;

    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        print_usage(&args[0]);
        std::process::exit(1);
    }
    let command = &args[1];
    let input = &args[2];
    let output = &args[3];

    // Dispatch based on command.
    let result = match command.as_str() {
        "--compress" | "-c" => compress_file(input, output),
        "--decompress" | "-d" => decompress_file(input, output),
        _ => {
            print_usage(&args[0]);
            return;
        }
    };

    if let Err(err) = result {
        eprintln!("{err}");
    }
}
