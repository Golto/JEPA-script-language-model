from pathlib import Path

from src.model import EncoderConfig, JEPAConfig, MLPPredictorConfig
from src.tokenizer import LanguageTokenizer
from src.training.pretrain import PretrainConfig, pretrain

config = PretrainConfig(
    data_dir=Path('.private/data/snippets'),
    output_dir=Path('.private/output/jepa'),
    jepa=JEPAConfig(
        encoder=EncoderConfig(
            vocab_size=LanguageTokenizer.VOCAB_SIZE,
            d_model=128,
            n_heads=4,
            n_layers=4,
            d_feedforward=512,
            dropout=0.1,
            max_seq_len=256,
            is_causal=False,
        ),
        predictor=MLPPredictorConfig(
            d_model=128,
            n_layers=2,
            d_hidden=128,
        ),
        ema_decay=0.999,
    ),
    n_blocks=3,
    n_epochs=50,
    batch_size=32,
    learning_rate=3e-4,
    weight_decay=1e-2,
    max_seq_len=256,
    log_every=10,
    device='cpu',
)

if __name__ == '__main__':
    pretrain(config)