include("networks.jl")
include("seq2seq.jl")
include("constants.jl")
include("discriminator.jl")

struct CycleGAN
    g_ab
    g_ba
    d_a
    d_b
end

@functor
function CycleGAN(
    g_input_dim::Int,
    g_output_dim::Int,
    pad_idx::Int,
    sos_idx::Int
) return CycleGAN(
    Seq2Seq(
        Encoder(g_input_dim, enc_emb_dim, g_hid_dim, g_n_layers, enc_dropout),
        Decoder(g_output_dim, dec_emb_dim, g_hid_dim, g_n_layers, dec_dropout),
        sos_idx
    ),
    Seq2Seq(
        Encoder(g_input_dim, enc_emb_dim, g_hid_dim, g_n_layers, enc_dropout),
        Decoder(g_output_dim, dec_emb_dim, g_hid_dim, g_n_layers, dec_dropout),
        sos_idx
    ),
    Discriminator(g_input_dim, d_emb_dim, pad_idx),
    Discriminator(g_input_dim, d_emb_dim, pad_idx)
)
end