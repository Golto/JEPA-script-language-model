"""Vocabulary definition for the embedding language tokenizer.

Defines the full token list, lookup tables, and special token ids
used by LanguageTokenizer for encoding and decoding.
"""

BOS_TOKEN = '<|sequence>'
EOS_TOKEN = '<sequence|>'
PAD_TOKEN = '<|padding|>'

VOCAB = [
    # Digits and decimal point
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.',
    # Booleans
    'true', 'false',
    # Registers
    'r0',  'r1',  'r2',  'r3',  'r4',  'r5',  'r6',  'r7',
    'r8',  'r9',  'r10', 'r11', 'r12', 'r13', 'r14', 'r15',
    # Arithmetic operators
    '+', '-', '*', '/', '%',
    # Logical operators
    'and', 'or', 'not',
    # Comparison operators
    '==', '!=', '<', '>', '<=', '>=',
    # Assignment and delimiters
    '=', '(', ')', ';', '\n',
    # Keywords
    'if', 'then', 'else', 'endif',
    'while', 'do', 'endwhile',
    'break', 'continue',
    'input', 'output',
    # Special tokens (completion)
    BOS_TOKEN, EOS_TOKEN, PAD_TOKEN,
]

TOKEN_TO_ID: dict[str, int] = {token: index for index, token in enumerate(VOCAB)}
ID_TO_TOKEN: dict[int, str] = {index: token for index, token in enumerate(VOCAB)}

VOCAB_SIZE = len(VOCAB)
BOS_ID = TOKEN_TO_ID[BOS_TOKEN]
EOS_ID = TOKEN_TO_ID[EOS_TOKEN]
PAD_ID = TOKEN_TO_ID[PAD_TOKEN]
