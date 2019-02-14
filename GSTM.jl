using Knet: @diff, Param, value, grad

using Statistics: norm, mean



mutable struct Layer
    wo1::Param
    wo2::Param
    wos::Param
    wk1::Param
    wk2::Param
    wks::Param
end

Layer(in_size1, in_size2, layer_size) =
begin
    sq  = sqrt(layer_size)
    wo1 = Param(randn(in_size1,layer_size)/sq)
    wo2 = Param(randn(in_size2,layer_size)/sq)
    wos = Param(randn(layer_size,layer_size)/sq)
    wk1 = Param(randn(in_size1,layer_size)/sq)
    wk2 = Param(randn(in_size2,layer_size)/sq)
    wks = Param(randn(layer_size,layer_size)/sq)
    layer = Layer(wo1, wo2, wos, wk1, wk2, wks)
layer
end

(layer::Layer)(state, in_1, in_2) =
begin
    out = tanh.(in_1 * layer.wo1 + in_2 * layer.wo2 + state * layer.wos)
    keep = sigmoid.(in_1 * layer.wk1 + in_2 * layer.wk2 + state * layer.wks)
    state = keep .* state + (1 .- keep) .* out
[out, state]
end



struct IS           # TODO: from here on make struct immutable?
    layer_1::Layer
    layer_2::Layer
    layer_out::Layer
end

IS(in_size1, in_size2, layer_sizes, out_size) =
begin
    is = IS(
        Layer(in_size1, in_size2, layer_sizes[1]),
        Layer(in_size1, layer_sizes[1], layer_sizes[2]),
        Layer(in_size1, layer_sizes[2], out_size),
    )
    state = [zeros(1, layer_sizes[1]),
             zeros(1, layer_sizes[2]),
             zeros(1, out_size)]
[is, state]
end

(is::IS)(states, in_1, in_2) =
begin
    out_1, state_1 = is.layer_1(states[1], in_1, in_2)
    out_2, state_2 = is.layer_2(states[2], in_1, out_1)
    out, state_out = is.layer_out(states[3], in_1, out_2)
[out, [state_1, state_2, state_out]]
end


struct GS
    layer_1::Layer
    layer_out::Layer
end

GS(in_size1, in_size2, layer_sizes, out_size) =
begin
    gs = GS(
        Layer(in_size1, in_size2, layer_sizes[1]),
        Layer(in_size1, layer_sizes[1], out_size),
    )
    state = [zeros(1, layer_sizes[1]),
             zeros(1, out_size)]
[gs, state]
end

(gs::GS)(states, in_1, in_2) =
begin
    out_1, state_1 = gs.layer_1(states[1], in_1, in_2)
    out, state_out = gs.layer_out(states[2], in_1, out_1)
[out, [state_1, state_out]]
end


struct GO
    layer_1::Layer
    layer_out::Layer
end

GO(in_size1, in_size2, layer_sizes, out_size) =
begin
    go = GO(
        Layer(in_size1, in_size2, layer_sizes[1]),
        Layer(in_size1, layer_sizes[1], out_size),
    )
    state = [zeros(1, layer_sizes[1]),
             zeros(1, out_size)]
[go, state]
end

(go::GO)(states, in_1, in_2) =
begin
    out_1, state_1 = go.layer_1(states[1], in_1, in_2)
    out, state_out = go.layer_out(states[2], in_1, out_1)
[out, [state_1, state_out]]
end





struct ENCODER
    is::IS
    gs::GS
    go::GO
end

ENCODER(hm_vectors, vector_size, storage_size, is_layers, gs_layers, go_layers) =
begin
    is, is_state = IS(storage_size, hm_vectors*vector_size, is_layers, storage_size)
    gs, gs_state = GS(storage_size, storage_size, gs_layers, storage_size)
    go, go_state = GO(storage_size, vector_size, go_layers, vector_size)
    encoder = ENCODER(is, gs, go)
    state   = [is_state,gs_state,[go_state for i in 1:hm_vectors]]
[encoder, state]
end

(encoder::ENCODER)(state, storage, vectors) =
begin
    is_out, is_state = encoder.is(state[1], storage, hcat(vectors...))
    gs_out, gs_state = encoder.gs(state[2], storage, is_out)
    go_out, go_state = [], []
    for (vector, state_v) in zip(vectors, state[3])
        go_out_v, go_state_v = encoder.go(state_v, is_out, vector)
        push!(go_out, go_out_v)
        push!(go_state, go_state_v)
    end
    output = [gs_out, go_out]
    state  = [is_state, gs_state, go_state]
[output, state]
end


struct DECODER
    is::IS
    gs::GS
    go::GO
end

DECODER(hm_vectors, vector_size, storage_size, is_layers, gs_layers, go_layers) =
begin
    is, is_state = IS(storage_size, hm_vectors*vector_size, is_layers, storage_size)
    gs, gs_state = GS(storage_size, storage_size, gs_layers, storage_size)
    go, go_state = GO(storage_size, vector_size, go_layers, vector_size)
    decoder = DECODER(is, gs, go)
    state   = [is_state,gs_state,[go_state for i in 1:hm_vectors]]
[decoder, state]
end

(decoder::DECODER)(state, storage, vectors, enc_storages, enc_vectors) =
begin
    attended = attend(storage, enc_storages, enc_vectors)
    is_out, is_state = decoder.is(state[1], storage, hcat(attended...))
    gs_out, gs_state = decoder.gs(state[2], storage, is_out)
    go_out, go_state = [], []
    for (vector, state_v) in zip(vectors, state[3])
        go_out_v, go_state_v = decoder.go(state_v, is_out, vector)
        push!(go_out, go_out_v)
        push!(go_state, go_state_v)
    end
    output = [gs_out, go_out]
    state  = [is_state, gs_state, go_state]
[output, state]
end



attend(storage, enc_storages, enc_vectors) =
begin
    similarities = [sum(storage .* enc_storage) for enc_storage in enc_storages]
enc_vectors[argmax(similarities)]
end


transfer_states!(enc_state, dec_state) =
begin
    for (dec_module, enc_module) in zip(dec_state, enc_state)
        for (dec_layer, enc_layer) in zip(dec_module, enc_module)
            i = 0
            for enc_neuron in enc_layer
                i +=1
                dec_layer[i] = enc_neuron
            end
        end
    end
end


propogate(enc, dec, enc_state, x, length_y; dec_state=nothing) =
begin
    enc_gs_time, enc_go_time = [], []
    enc_outs = [zeros(1, storage_size), nothing]
    for timestep in x
        enc_outs, enc_state = enc(enc_state, enc_outs[1], timestep)
        push!(enc_gs_time, enc_outs[1])
        push!(enc_go_time, enc_outs[2])
    end

    # transfer_states!(enc_state, dec_state)

    dec_go_time = []
    if dec_state == nothing dec_state = enc_state end
    dec_outs = enc_outs # [zeros(1, storage_size), [zeros(1, vector_size) for i in 1:hm_vectors]]
    for timestep in 1:length_y
        dec_outs, dec_state = dec(dec_state, dec_outs[1], dec_outs[2], enc_gs_time, enc_go_time)
        push!(dec_go_time, dec_outs[2])
    end
dec_go_time
end


upd!(encoder, decoder, result, lr) =
begin
    for model in [encoder, decoder]
        model_fields = fieldnames(typeof(model))
        for network in [getfield(model, field) for field in model_fields]
            network_fields = fieldnames(typeof(network))
            for layer in [getfield(network, field) for field in network_fields]
                layer_fields = fieldnames(typeof(layer))
                for field in layer_fields
                    value = getfield(layer, field)
                    setfield!(layer, field, Param(value - grad(result, value) .*lr))
                end
            end
        end
    end
end


sigmoid(x)  = 1 / (1+exp(-x))

softmax(x)  = norm(exp(x))

mse(out, y) = mean((out .- y) .^2)


sequence_loss(out, y) =
begin
    sum([sum(sum([mse(out_e, y_e) for (out_e,y_e) in zip(out_t,y_t)])) for (out_t,y_t) in zip(out, y)])
end
