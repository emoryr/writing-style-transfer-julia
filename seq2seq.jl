include("networks.jl")

struct Seq2Seq
    encoder
    decoder
    sos_idx
end

@functor
function Seq2Seq(
    encoder::Encoder,
    decoder::Decoder,
    sos_idx::Int
) return Seq2Seq(
    encoder,
    decoder,
    sos_idx
)
end

function(net::Seq2Seq)(x, teacher_forcing_ratio::Float32 = 0.5L)
    batch_size = size(x)[1]
    max_len = size(x)[0]
    src_vocab_size = net.decoder.output_dim

    outputs = zeros(max_len, batch_size, src_vocab_size)
    max_output = zeros(max_len, batch_size)

    hidden, cell = net.encoder(x)

    input_data = view(x,[0:end])
    max_output[0] = input_dat

    hidden, cell = net.encoder(src)

    for t in range(2, max_len)
        output, hidden, cell = net.decoder(input_data, hidden, cell)
        outputs[t] = output
        top = output.max(1)[1]
        max_output[t] = top

        if random.random() < teacher_forcing_ratio
            input_data = src[t]
        else
            input_data = top
        end
    end

    return outputs, max_output
end