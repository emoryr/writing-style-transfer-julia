using Flux: Embedding, LSTM
using Functors

struct Decoder
    enbedding
    rnn
    out
    dropout
end

@functor Decoder
function Decoder(output_dim::Int,
    emb_dim::Int,
    hid_dim::Int,
    n_layers::Int,
    dropout::Int)

    layers = []
    for index in range(1, n_layers - 1)
        push!(layers, LSTM(emb_dim, emb_dim))
    end

    push!(layers, LSTM(emb_dim, hid_dim))

    return Decoder(
        Embedding(output_dim, emb_dim),
        Chain(layers...,
            Dropout(dropout)),
        Dense(hid_dim, output_dim),
        Dropout(dropout)
    )
end

function (net::Decoder)(x)

    input_data = unsqueeze!(x, 0)

    embedded = net.dropout(net.embedding(input_data))

    output, (hidden, cell) = net.rnn(embedded, (hidden, cell))

    prediction = net.out(squeeze(output, 0))

    return prediction, hidden, cell
end


struct Encoder
    enbedding
    rnn
    dropout
end

@functor Encoder
function Encoder(output_dim::Int,
    emb_dim::Int,
    hid_dim::Int,
    n_layers::Int,
    dropout::Int)

    layers = []
    for index in range(1,n_layers -1)
        push!(layers, LSTM(emb_dim, emb_dim))
    end

    push!(layers, LSTM(emb_dim, hid_dim))

    return Encoder(
        Embedding(output_dim, emb_dim),
        Chain(layers...,
            Dropout(dropout)),
        Dropout(dropout)
    )
end

function (net::Encoder)(x)
    embedded = net.dropout(net.embedding(x))

    outputs, (hidden, cell) = self.rnn(embedded)
    return hidden, outputs
end
