using Flux
using Functors
using Images

struct Discriminator
    embedding
    fc
end

@functor
function Discriminator(
    vocab_size::Int,
    embedding_dim::Int,
    pad_idx::Int

) return Discriminator(
    Chain(Embedding(vocab_size, embedding_dim),
        Pad((pad_idx, pad_idx))),
    Dense(embedding_dim, 1)
)
end

function(net::Discriminator)(x)
    embedded = net.embedding(x)

    embedded = permutedims(embedded,(2, 1, 3))

    pooled = squeeze(meanpool(embedded, (size(embedded)[2], 1)), 2)

    return net.fc(pooled)
end