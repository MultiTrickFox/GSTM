include("GSTM.jl")


hm_vectors   = 4
vector_size  = 13
storage_size = 20

is_layers = [50, 35]
gs_layers = [30]
go_layers = [30]


make_model(hm_vectors, vector_size, storage_size, is_layers, gs_layers, go_layers) =
begin
    enc, enc_states = ENCODER(hm_vectors, vector_size, storage_size, is_layers, gs_layers, go_layers)
    dec, dec_states = DECODER(hm_vectors, vector_size, storage_size, is_layers, gs_layers, go_layers)
    model = [enc,dec]
    state = [enc_states,dec_states]
[model, state]
end

(encoder, decoder), (enc_zero_state, dec_zero_state) = make_model(hm_vectors, vector_size, storage_size, is_layers, gs_layers, go_layers)

make_data(hm_data) =
begin
    data = []
    for i in 1:hm_data
        push!(data,
            [
                [[randn(1, vector_size) for iii in 1:hm_vectors] for ii in 1:(rand()+1)*75],
                [[randn(1, vector_size) for iii in 1:hm_vectors] for ii in 1:(rand()+1)*75]
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


function train(data, lr, epochs)

    for epoch in 1:epochs

        loss = 0

        for (x,y) in shuffle(data)

            # sequence_loss(propogate(encoder, decoder, enc_zero_state, dec_zero_state, x, y), y)

            result = @diff sequence_loss(
                        propogate(encoder, decoder, enc_zero_state, x, length(y)),
                        y
                    )

            loss += value(result)

            println(value(result))

            upd!(encoder, decoder, result, lr)

            print("/")
        end
        println(" ")

        println("Epoch ", epoch, " Loss ", loss)

    end

end


@time train(make_data(20), .001, 10)
