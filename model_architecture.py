from transformer import Transformer
import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.load('urdu_tokenizer.model')

vocab_size = sp.vocab_size()
# Create the model
model = Transformer(
    vocab_size=vocab_size,
    d_model=256,
    num_heads=2,
    num_encoder_layers=2,
    num_decoder_layers=2,
    d_ff=1024,
    dropout=0.1,
    max_len=512
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
# Count total parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Model initialized!")
print(f"Device: {device}")
print(f"Vocabulary size: {vocab_size}")
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
# Making a dummy input
batch_size = 2
src_len = 10
tgt_len = 8

src = torch.randint(0, vocab_size, (batch_size, src_len)).to(device)
tgt = torch.randint(0, vocab_size, (batch_size, tgt_len)).to(device)

# Forward pass
with torch.no_grad():
    output = model(src, tgt)
print("Model forward pass successful!")
print(f"Input shape (src): {src.shape}")
print(f"Input shape (tgt): {tgt.shape}")
print(f"Output shape: {output.shape}")
print(f"Expected: ({batch_size}, {tgt_len}, {vocab_size})")
