import streamlit as st
import torch
import torch.nn.functional as F
import sentencepiece as spm
from transformer import Transformer

# -------------------------------------------------------
# Load model + tokenizer ONLY ONCE
# -------------------------------------------------------
@st.cache_resource
def load_model_and_tokenizer():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer
    sp = spm.SentencePieceProcessor()
    sp.load("urdu_tokenizer.model")

    # Load model (Transformer class from transformer.py)
    model = Transformer(vocab_size=sp.vocab_size())
    model.load_state_dict(torch.load("best_masked_model.pt", map_location=device))
    model.to(device)
    model.eval()

    return model, sp, device

model, sp, device = load_model_and_tokenizer()

# -------------------------------------------------------
# Decoding Functions
# -------------------------------------------------------
def greedy_decode(model, src, sp, device, max_len=50):
    model.eval()
    src_mask = (src != 0).unsqueeze(1).unsqueeze(2)

    with torch.no_grad():
        enc_output = model.encoder_embedding(src) * (model.d_model ** 0.5)
        enc_output = model.positional_encoding(enc_output)
        for layer in model.encoder_layers:
            enc_output = layer(enc_output, src_mask)

    ys = torch.ones(1, 1, dtype=torch.long, device=device) * 2  # BOS token

    for _ in range(max_len - 1):
        tgt_mask = torch.tril(torch.ones((1, ys.size(1), ys.size(1)), device=device)).bool()
        out = model.decoder_embedding(ys) * (model.d_model ** 0.5)
        out = model.positional_encoding(out)
        dec_output = out
        for layer in model.decoder_layers:
            dec_output = layer(dec_output, enc_output, src_mask, tgt_mask)
        logits = model.fc_out(dec_output[:, -1])
        next_token = torch.argmax(logits, dim=-1).item()
        if next_token == 3:  # EOS token
            break
        ys = torch.cat([ys, torch.tensor([[next_token]], device=device)], dim=1)
    return sp.decode_ids(ys.squeeze().tolist()[1:])

def beam_search_decode(model, src, sp, device, beam_width=3, max_len=50):
    model.eval()
    src_mask = (src != 0).unsqueeze(1).unsqueeze(2)

    with torch.no_grad():
        enc_output = model.encoder_embedding(src) * (model.d_model ** 0.5)
        enc_output = model.positional_encoding(enc_output)
        for layer in model.encoder_layers:
            enc_output = layer(enc_output, src_mask)

    beams = [(torch.tensor([[2]], device=device), 0)]  # (sequence, score)
    for _ in range(max_len):
        new_beams = []
        for seq, score in beams:
            tgt_mask = torch.tril(torch.ones((1, seq.size(1), seq.size(1)), device=device)).bool()
            out = model.decoder_embedding(seq) * (model.d_model ** 0.5)
            out = model.positional_encoding(out)
            dec_output = out
            for layer in model.decoder_layers:
                dec_output = layer(dec_output, enc_output, src_mask, tgt_mask)
            logits = model.fc_out(dec_output[:, -1])
            probs = F.log_softmax(logits, dim=-1).squeeze(0)
            topk = torch.topk(probs, beam_width)
            for i in range(beam_width):
                next_tok = topk.indices[i].item()
                next_score = score + topk.values[i].item()
                new_seq = torch.cat([seq, torch.tensor([[next_tok]], device=device)], dim=1)
                new_beams.append((new_seq, next_score))
        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
        if all(seq[0, -1].item() == 3 for seq, _ in beams):
            break
    best_seq = beams[0][0].squeeze().tolist()
    best_seq = [t for t in best_seq[1:] if t not in [0, 3]]
    return sp.decode_ids(best_seq)

def top_k_sampling(logits, k=50, temperature=0.9):
    logits = logits / temperature
    values, indices = torch.topk(logits, k)
    probs = F.softmax(values, dim=-1)
    return indices[torch.multinomial(probs, 1)].item()

def top_k_decode(model, src, sp, device, k=50, temperature=0.9, max_len=50):
    model.eval()
    src_mask = (src != 0).unsqueeze(1).unsqueeze(2)

    with torch.no_grad():
        enc_output = model.encoder_embedding(src) * (model.d_model ** 0.5)
        enc_output = model.positional_encoding(enc_output)
        for layer in model.encoder_layers:
            enc_output = layer(enc_output, src_mask)

    ys = torch.ones(1, 1, dtype=torch.long, device=device) * 2
    for _ in range(max_len - 1):
        tgt_mask = torch.tril(torch.ones((1, ys.size(1), ys.size(1)), device=device)).bool()
        out = model.decoder_embedding(ys) * (model.d_model ** 0.5)
        out = model.positional_encoding(out)
        dec_output = out
        for layer in model.decoder_layers:
            dec_output = layer(dec_output, enc_output, src_mask, tgt_mask)
        logits = model.fc_out(dec_output[:, -1])
        next_token = top_k_sampling(logits.squeeze(), k=k, temperature=temperature)
        if next_token == 3:
            break
        ys = torch.cat([ys, torch.tensor([[next_token]], device=device)], dim=1)
    return sp.decode_ids(ys.squeeze().tolist()[1:])

def generate_response(model, src_text, sp, device, decoding="Greedy", max_len=50):
    src_tokens = [2] + sp.encode_as_ids(src_text) + [3]
    src = torch.LongTensor([src_tokens]).to(device)
    if decoding == "Greedy":
        return greedy_decode(model, src, sp, device, max_len)
    elif decoding == "Beam Search":
        return beam_search_decode(model, src, sp, device, max_len=max_len)
    elif decoding == "Top-k Sampling":
        return top_k_decode(model, src, sp, device, max_len=max_len)

# -------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------
st.set_page_config(page_title="Urdu Chatbot", layout="wide")
st.markdown("""
## 🇵🇰 Urdu Chatbot — Transformer Encoder-Decoder  
### **Decoding: Greedy | Beam Search | Top-k Sampling**  
Right-to-left Urdu text supported.
""")

# Maintain conversation history
if "history" not in st.session_state:
    st.session_state.history = []

col1, col2 = st.columns(2)

with col1:
    user_input = st.text_area(
        "📝 پیغام لکھیں",
        placeholder="یہاں لکھیں...",
        height=120
    )
    decoding_strategy = st.radio(
        "Decoding Strategy",
        ["Greedy", "Beam Search", "Top-k Sampling"],
        index=2
    )
    if st.button("Send ➤"):
        if user_input.strip():
            reply = generate_response(model, user_input, sp, device, decoding_strategy)
            st.session_state.history.append(f"آپ: {user_input}")
            st.session_state.history.append(f"بوٹ: {reply}")
            st.success(reply)
        else:
            st.error("براہ کرم کوئی پیغام لکھیں۔")

with col2:
    st.text_area(
        "گفتگو کا خلاصہ",
        value="\n".join(st.session_state.history),
        height=380
    )
