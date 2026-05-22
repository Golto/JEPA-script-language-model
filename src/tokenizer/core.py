from src.language.lexer import Lexer
from src.language.lexer.tokens import TokenType
from .vocabulary import (
    TOKEN_TO_ID,
    ID_TO_TOKEN,
    VOCAB_SIZE,
    BOS_TOKEN,
    EOS_TOKEN,
    BOS_ID,
    EOS_ID,
    PAD_ID,
)


DIGIT_CHARS: frozenset[str] = frozenset('0123456789.')


class UnknownTokenError(Exception):
    """Raised when a token or character is not present in the vocabulary."""

    def __init__(self, value: str):
        super().__init__(f"Unknown token in vocabulary: '{value}'")


class LanguageTokenizer:
    """Tokenizer for the embedding language.

    Encodes a source program into a sequence of integer ids.
    Numbers are decomposed digit by digit, with an optional decimal point.

    Supports two encoding modes:
        - completion: encode()           produces BOS code EOS
        - instruct:   encode_instruct()  produces BOS spec code EOS
    """

    VOCAB_SIZE = VOCAB_SIZE
    BOS_ID = BOS_ID
    EOS_ID = EOS_ID
    PAD_ID = PAD_ID

    # ----------------------------------------------------------------
    # Encoding
    # ----------------------------------------------------------------

    def encode(self, source: str, add_special_tokens: bool = True) -> list[int]:
        """Encode a source program into a list of token ids.

        Args:
            source: Raw source code string.
            add_special_tokens: Whether to prepend BOS and append EOS ids.

        Returns:
            A list of integer token ids.
        """
        tokens = self._lex(source)
        ids: list[int] = []
        for token in tokens:
            ids.extend(self._token_to_ids(token))
        if add_special_tokens:
            ids = [BOS_ID] + ids + [EOS_ID]
        return ids

    # ----------------------------------------------------------------
    # Lexing and token conversion
    # ----------------------------------------------------------------

    def _lex(self, source: str) -> list[str]:
        """Tokenize source code using the language lexer.

        EOF tokens are dropped. NEWLINE tokens are normalized to '\\n'.

        Args:
            source: Raw source code string.

        Returns:
            A list of token value strings.
        """
        lexer = Lexer(source)
        token_values: list[str] = []
        for token in lexer.tokenize():
            if token.type == TokenType.EOF:
                continue
            if token.type == TokenType.NEWLINE:
                token_values.append('\n')
                continue
            token_values.append(token.value)
        return token_values

    def _token_to_ids(self, value: str) -> list[int]:
        """Convert a single lexical value into a list of vocabulary ids.

        Numbers (INTEGER or FLOAT) are decomposed character by character.

        Args:
            value: A lexical token value string.

        Returns:
            A list of one or more vocabulary ids.

        Raises:
            UnknownTokenError: If any character or token is absent from the vocabulary.
        """
        if all(character in DIGIT_CHARS for character in value):
            ids: list[int] = []
            for character in value:
                if character not in TOKEN_TO_ID:
                    raise UnknownTokenError(character)
                ids.append(TOKEN_TO_ID[character])
            return ids

        if value not in TOKEN_TO_ID:
            raise UnknownTokenError(value)
        return [TOKEN_TO_ID[value]]

    # ----------------------------------------------------------------
    # Decoding
    # ----------------------------------------------------------------

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> list[str]:
        """Decode a list of token ids back into token strings.

        Args:
            ids: List of integer vocabulary ids.
            skip_special_tokens: Whether to omit BOS and EOS tokens from output.

        Returns:
            A list of token strings.

        Raises:
            UnknownTokenError: If an id has no corresponding vocabulary entry.
        """
        special_tokens = {BOS_TOKEN, EOS_TOKEN}
        result: list[str] = []
        for token_id in ids:
            if token_id not in ID_TO_TOKEN:
                raise UnknownTokenError(f"id={token_id}")
            token = ID_TO_TOKEN[token_id]
            if skip_special_tokens and token in special_tokens:
                continue
            result.append(token)
        return result

    def tokens_to_source(
        self,
        tokens: list[str],
        indent: int | None = 4,
    ) -> str:
        """Reconstruct a readable source program from a list of token strings.

        Reconstruction rules:
        - Consecutive digit and '.' tokens are merged into a single number.
        - A '-' immediately before a digit group is treated as a unary sign
          when the preceding token is an operator, delimiter, or keyword.
        - Newline tokens ('\\n') are emitted as-is.
        - Indentation level is driven by then/do/else/endif/endwhile tokens.
        - A space is inserted between tokens except before '\\n'.

        Args:
            tokens: A list of token strings as returned by decode().
            indent: Number of spaces per indentation level. None disables indentation.

        Returns:
            A reconstructed source code string.
        """
        unary_trigger_tokens: set[str | None] = {
            '=', '(', '+', '-', '*', '/', '%',
            '==', '!=', '<', '>', '<=', '>=',
            'and', 'or', 'not',
            'then', 'do', 'else', 'if', 'while',
            'input', 'output', 'return',
            '\n', None,
        }
        spec_line_starter_tokens: set[str] = {'example', 'input-type', 'output-type'}
        digit_chars_no_dot: frozenset[str] = DIGIT_CHARS - {'.'}

        parts: list[str] = []
        current_indent = 0
        is_new_line = True
        is_in_spec_line = False
        prev_token: str | None = None

        token_index = 0
        while token_index < len(tokens):
            token = tokens[token_index]

            # ----------------------------------------------------------------
            # Newline
            # ----------------------------------------------------------------
            if token == '\n':
                if parts and parts[-1] == ' ':
                    parts.pop()
                parts.append('\n')
                is_new_line = True
                is_in_spec_line = False
                prev_token = '\n'
                token_index += 1
                continue

            # Dedent before writing endif/endwhile/else
            if indent is not None and token in ('endif', 'endwhile', 'else'):
                current_indent = max(0, current_indent - 1)

            # Emit indentation at the start of a new line
            if indent is not None and is_new_line:
                parts.append(' ' * (current_indent * indent))
                is_new_line = False

            if token in spec_line_starter_tokens:
                is_in_spec_line = True

            is_unary_minus = (
                token == '-'
                and (token_index + 1) < len(tokens)
                and tokens[token_index + 1] in digit_chars_no_dot
                and (is_in_spec_line or prev_token in unary_trigger_tokens)
            )

            # ----------------------------------------------------------------
            # Number assembly (with optional leading unary minus)
            # ----------------------------------------------------------------
            if token in DIGIT_CHARS or is_unary_minus:
                number_buffer = ''
                if is_unary_minus:
                    number_buffer += '-'
                    token_index += 1

                has_dot = False
                while token_index < len(tokens) and tokens[token_index] in DIGIT_CHARS:
                    character = tokens[token_index]
                    if character == '.':
                        if has_dot:
                            break
                        has_dot = True
                    number_buffer += character
                    token_index += 1

                parts.append(number_buffer)
                prev_token = number_buffer

            # ----------------------------------------------------------------
            # Regular token
            # ----------------------------------------------------------------
            else:
                parts.append(token)
                prev_token = token
                token_index += 1

            # Indent after then/do/else
            if indent is not None and token in ('then', 'do', 'else'):
                current_indent += 1

            # Insert space separator unless the next token is a newline
            if token_index < len(tokens) and tokens[token_index] != '\n':
                parts.append(' ')

        return ''.join(parts)
