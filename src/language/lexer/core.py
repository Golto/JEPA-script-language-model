from typing import List, Optional
from .tokens import Token, TokenType


# ─── Exception ────────────────────────────────────────────────────────────────

class LanguageLexicalError(Exception):
    """Erreur lexicale détectée pendant l'analyse lexicale"""
    def __init__(self, message: str, line: int, column: int):
        self.message = message
        self.line = line
        self.column = column
        super().__init__(f"Erreur lexicale à la ligne {line}, colonne {column}: {message}")


# ─── Analyseur lexical ────────────────────────────────────────────────────────

class Lexer:

    def __init__(self, source: str):
        self.source = source
        self.position = 0
        self.line = 1
        self.column = 1
        self.current_char: Optional[str] = self.source[0] if source else None

        self.keywords = {
            # Data types
            'true': TokenType.BOOLEAN,
            'false': TokenType.BOOLEAN,
            # Registers
            'r0': TokenType.REGISTER,
            'r1': TokenType.REGISTER,
            'r2': TokenType.REGISTER,
            'r3': TokenType.REGISTER,
            'r4': TokenType.REGISTER,
            'r5': TokenType.REGISTER,
            'r6': TokenType.REGISTER,
            'r7': TokenType.REGISTER,
            'r8': TokenType.REGISTER,
            'r9': TokenType.REGISTER,
            'r10': TokenType.REGISTER,
            'r11': TokenType.REGISTER,
            'r12': TokenType.REGISTER,
            'r13': TokenType.REGISTER,
            'r14': TokenType.REGISTER,
            'r15': TokenType.REGISTER,
            # Logical operators
            'and': TokenType.AND,
            'or': TokenType.OR,
            'not': TokenType.NOT,
            # Keywords
            'if': TokenType.IF,
            'then': TokenType.THEN,
            'else': TokenType.ELSE,
            'endif': TokenType.ENDIF,
            'while': TokenType.WHILE,
            'do': TokenType.DO,
            'endwhile': TokenType.ENDWHILE,
            'break': TokenType.BREAK,
            'continue': TokenType.CONTINUE,
            # I/O
            'input': TokenType.INPUT,
            'output': TokenType.OUTPUT
        }


    # ─── Méthodes communes ────────────────────────────────────────────────────
    
    def advance(self):
        """Avance d'un caractère dans le code source"""
        if self.current_char == '\n':
            self.line += 1
            self.column = 1
        else:
            self.column += 1
        
        self.position += 1
        self.current_char = self.source[self.position] if self.position < len(self.source) else None


    def peek(self, offset: int = 1) -> Optional[str]:
        """Regarde le caractère à une position future sans avancer"""
        peek_position = self.position + offset
        return self.source[peek_position] if peek_position < len(self.source) else None
    
    
    # ─── Méthodes de Parsing ──────────────────────────────────────────────────

    def skip_whitespace(self):
        """Ignore les espaces et tabulations (mais pas les retours à la ligne)"""
        while self.current_char and self.current_char in ' \t\r':
            self.advance()

    
    def read_number(self) -> Token:
        """Lit un nombre (entier, flottant ou complexe)"""
        start_line = self.line
        start_column = self.column
        num_str = ''
        has_dot = False
        
        while self.current_char and (self.current_char.isdigit() or self.current_char in '._'):
            if self.current_char == '.':
                if has_dot:
                    break
                has_dot = True
                num_str += self.current_char
                self.advance()
            elif self.current_char == '_':
                # Permet les underscores dans les nombres (ex: 1_000_000)
                self.advance()
            else:
                num_str += self.current_char
                self.advance()
        
        if has_dot:
            return Token(TokenType.FLOAT, num_str, start_line, start_column)
        else:
            return Token(TokenType.INTEGER, num_str, start_line, start_column)
    
    
    # ─── Méthodes de tokénisation ─────────────────────────────────────────────

    def get_next_token(self) -> Token:
        """Retourne le prochain token"""
        while self.current_char:
            # Ignore les espaces
            if self.current_char in ' \t\r':
                self.skip_whitespace()
                continue

            # Nouvelle ligne
            if self.current_char == '\n':
                token = Token(TokenType.NEWLINE, '\\n', self.line, self.column)
                self.advance()
                return token
            
            # Nombres
            if self.current_char.isdigit():
                return self.read_number()
            
            # Registres et mots-clés
            if self.current_char.isalpha() or self.current_char == '_':
                start_line, start_col = self.line, self.column
                word = ''
                while self.current_char and (self.current_char.isalnum() or self.current_char == '_'):
                    word += self.current_char
                    self.advance()
                
                token_type = self.keywords.get(word)
                if token_type is None:
                    raise LanguageLexicalError(f"Identifiant inconnu '{word}'", start_line, start_col)
                
                return Token(token_type, word, start_line, start_col)
            
            # Opérateurs et délimiteurs
            line, col = self.line, self.column
            char = self.current_char
            
            # Opérateurs de comparaison
            if char == '=' and self.peek() == '=':
                self.advance()
                self.advance()
                return Token(TokenType.EQ, '==', line, col)
            
            if char == '!' and self.peek() == '=':
                self.advance()
                self.advance()
                return Token(TokenType.NEQ, '!=', line, col)
            
            if char == '<' and self.peek() == '=':
                self.advance()
                self.advance()
                return Token(TokenType.LTE, '<=', line, col)
            
            if char == '>' and self.peek() == '=':
                self.advance()
                self.advance()
                return Token(TokenType.GTE, '>=', line, col)
            
            # Opérateurs simples
            single_char_tokens = {
                '+': TokenType.ADD,
                '-': TokenType.SUB,
                '*': TokenType.MUL,
                '/': TokenType.DIV,
                '%': TokenType.MOD,
                '=': TokenType.ASSIGN,
                '<': TokenType.LT,
                '>': TokenType.GT,
                '(': TokenType.LPAREN,
                ')': TokenType.RPAREN,
                ';': TokenType.SEMICOLON
            }
            
            if char in single_char_tokens:
                token_type = single_char_tokens[char]
                self.advance()
                return Token(token_type, char, line, col)

            raise LanguageLexicalError(f"Caractère inattendu '{char}'", line, col)

        return Token(TokenType.EOF, '', self.line, self.column)


    def tokenize(self) -> List[Token]:
        """Tokenise tout le code source"""
        tokens = []
        token = self.get_next_token()
        
        while token.type != TokenType.EOF:
            # Ignore les nouvelles lignes multiples
            if token.type == TokenType.NEWLINE:
                if not tokens or tokens[-1].type != TokenType.NEWLINE:
                    tokens.append(token)
            else:
                tokens.append(token)
            token = self.get_next_token()
        
        tokens.append(token)  # Add EOF
        return tokens
    
    # ─── Méthodes Helpers ─────────────────────────────────────────────────────

    def __repr__(self):
        return f"<Lexer line={self.line} column={self.column} position={self.position}>"