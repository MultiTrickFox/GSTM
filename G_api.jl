using Distributed: @everywhere, @distributed, addprocs; addprocs(3)
@everywhere include("GSTM_v2.jl")



const hm_vectors   = 4
const vector_size  = 13
const storage_size = 25


const is_layers = [85]
const gs_layers = [45]
const go_layers = [55]


const max_t     = 100
const hm_data   = 50

const lr        = 1e-5
const hm_epochs = 250



make_model(hm_vectors, vector_size, storage_size, is_layers, gs_layers, go_layers) =
begin
    enc, enc_states = ENCODER(hm_vectors, vector_size, storage_size, is_layers, gs_layers, go_layers)
    dec, dec_states = DECODER(hm_vectors, vector_size, storage_size, is_layers, gs_layers, go_layers)
    model = [enc,dec]
    state = [enc_states,dec_states]
[model, state]
end

(encoder, decoder), (enc_zerostate, dec_zerostate) = make_model(hm_vectors, vector_size, storage_size, is_layers, gs_layers, go_layers)

make_data(hm_data; max_timesteps=50) =
begin
    data = []
    for i in 1:hm_data
        push!(data,
            [
                [[randn(1, vector_size) for iii in 1:hm_vectors] for ii in 1:rand(max_timesteps/2:max_timesteps)],
                [[randn(1, vector_size) for iii in 1:hm_vectors] for ii in 1:rand(max_timesteps/2:max_timesteps)]
            ]
        )
    end
data
end

make_accugrads(hm_vectors, vector_size, storage_size, is_layers, gs_layers, go_layers) =
begin
    accu_grads = []
    for model in [encoder, decoder]
        for mfield in fieldnames(typeof(model))
            net = getfield(model, mfield)
            for nfield in fieldnames(typeof(net))
                layer = getfield(net, nfield)
                for lfield in fieldnames(typeof(layer))
                    field_size = size(getfield(layer, lfield))
                    push!(accu_grads, zeros(field_size))
                end
            end
        end
    end
accu_grads
end


shuffle(arr_in) =
begin
    array_copy = copy(arr_in)
    array_new  = []
    while length(array_copy) > 0
        index = rand(1:length(array_copy))
        e = array_copy[index]
        deleteat!(array_copy, index)
        push!(array_new, e)
    end
array_new
end

accu_grads = make_accugrads(hm_vectors, vector_size, storage_size, is_layers, gs_layers, go_layers)


train(data, (encoder, decoder), enc_zerostate, lr, ep; accu_grads=nothing) =
begin
    @show lr

    losses = []

    for epoch in 1:ep

        print("Epoch ", epoch, ": ")

        loss = 0.0
        prev_loss = 0.0

        for (g,l) in

            (@distributed (vcat) for (x,y) in shuffle(data)

                d = @diff sequence_loss(
                    propogate(encoder, decoder, enc_zerostate, x, length(y)),
                    y
                )

                grads(d, encoder, decoder), value(d)

            end)

            upd!(encoder, decoder, g, lr)
            #upd_rms!(encoder, decoder, g, lr, accu_grads, alpha=.9)
            loss += sum(l)

        end ; @show loss ; push!(losses, loss)

    end
[encoder, decoder, losses]
end



@time (enc, dec, loss) =

    train(make_data(hm_data,
           max_timesteps=max_t),
         (encoder, decoder),
          enc_zerostate,
          lr,
          hm_epochs,
          accu_grads=accu_grads)


# using PyPlot: plot
# plot(loss, collect(1:hm_epochs), color="red", linewidth=2.0, linestyle="--")
