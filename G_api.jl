using Distributed: @everywhere, @distributed, addprocs; addprocs(20)
@everywhere include("GSTM.jl")



const hm_vectors   = 4
const vector_size  = 13
const storage_size = 25


const is_layers = [40, 30]
const gs_layers = [30]
const go_layers = [35]



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
                [[randn(1, vector_size) for iii in 1:hm_vectors] for ii in 1:rand(5:max_timesteps)],#(rand()+1)*max_timestep],
                [[randn(1, vector_size) for iii in 1:hm_vectors] for ii in 1:rand(5:max_timesteps)]#(rand()+1)*max_timestep]
            ]
        )
    end
data
end


shuffle!(arr_in) =
begin
    hm = length(arr_in)
    for i in hm
      ph = arr_in[i]
      loc = rand(1:hm)
      arr_in[i] = arr_in[loc]
      arr_in[loc] = ph
    end
arr_in
end


train(data, (encoder, decoder), enc_zerostate, lr, ep) =

    for epoch in 1:ep

        println("Epoch ", epoch, ": ")

        batch_size = length(data)

        loss = 0.0

        for (g,l) in

            (@distributed (vcat) for (x,y) in shuffle!(data)

                d = @diff sequence_loss(
                    propogate(encoder, decoder, enc_zerostate, x, length(y)),
                    y
                )

                grads(d, encoder, decoder), value(d)

            end)

            upd!(encoder, decoder, g, lr, batch_size)
            loss += l
        end

        @show sum(loss)
    end



@time train(make_data(
                100, max_timesteps=50),
        (encoder, decoder),
         enc_zerostate,
         .0001,
         20)
