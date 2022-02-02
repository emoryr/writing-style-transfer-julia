using Flux
using Functors

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
    Embedding(vocab_size, embedding_dim),
    Dense(embedding_dim, 1)
)
end

function(net::Discriminator)(x)
    return
end