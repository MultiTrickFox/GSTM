using Distributed: @everywhere, @distributed, addprocs; addprocs(3)
@everywhere include("GSTM.jl")


const hm_vectors   = 4
const vector_size  = 13
const storage_size = 20

const is_layers = [50, 35]
const gs_layers = [30]
const go_layers = [30]



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
                [[randn(1, vector_size) for iii in 1:hm_vectors] for ii in 1:max_timesteps],#(rand()+1)*max_timestep],
                [[randn(1, vector_size) for iii in 1:hm_vectors] for ii in 1:max_timesteps]#(rand()+1)*max_timestep]
            ]
        )
    end
data
end


function shuffle(arr_in)
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


train(data, (encoder, decoder), enc_zerostate, lr, ep) =

    for epoch in 1:ep

        println("Epoch ", epoch, ": ")

        loss = 0.0

        for (g,l) in

            (@distributed (vcat) for (x,y) in shuffle(data)

                d = @diff sequence_loss(
                    propogate(encoder, decoder, deepcopy(enc_zerostate), x, length(y), dec_state=deepcopy(dec_zerostate)),
                    y
                )

                print("/")

                grads(d, encoder, decoder), value(d)

            end)

            upd!(encoder, decoder, g, lr)
            loss += l
        end

        @show sum(loss)
    end



@time train(make_data(
                100, max_timesteps=50),
        (encoder, decoder),
         enc_zerostate,
         .001,
         12)
