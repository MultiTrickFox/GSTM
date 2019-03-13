using Knet: @diff, Param, value, grad
using Knet: sigm, tanh

relu(x) = max(0,x)



mutable struct Layer
    wf1::Param
    wf2::Param
    wfs::Param
    wi1::Param
    wi2::Param
    wk1::Param
    wk2::Param
    wks::Param
end

Layer(in_size1, in_size2, layer_size) =
begin
    sq = sqrt(2/(in_size1+in_size2+layer_size))
    wf1 = Param(2*sq .* randn(in_size1,layer_size)   .-sq)
    wf2 = Param(2*sq .* randn(in_size2,layer_size)   .-sq)
    wfs = Param(2*sq .* randn(layer_size,layer_size) .-sq)
    wi1 = Param(2*sq .* randn(in_size1,layer_size)   .-sq)
    wi2 = Param(2*sq .* randn(in_size2,layer_size)   .-sq)
    wk1 = Param(2*sq .* randn(in_size1,layer_size)   .-sq)
    wk2 = Param(2*sq .* randn(in_size2,layer_size)   .-sq)
    wks = Param(2*sq .* randn(layer_size,layer_size) .-sq)
    layer = Layer(wf1, wf2, wfs, wi1, wi2, wk1, wk2, wks)
layer
end

(layer::Layer)(state, in_1, in_2) =
begin
    attention = sigm.(in_1 * layer.wf1 + in_2 * layer.wf2 + state * layer.wfs)
    remember  = sigm.(in_1 * layer.wk1 + in_2 * layer.wk2 + state * layer.wks)
    interm    = tanh.(in_1 * layer.wi1 + in_2 * layer.wi2 + state .* attention)
    state = remember .* state + (1 .- remember) .* interm
[state, state]
end





struct IS
    layer_1::Layer
    layer_out::Layer
end

IS(storage_size, hm_vectors, vector_size, layer_sizes, out_size) =
begin
    is = IS(
        Layer(storage_size, hm_vectors*vector_size, layer_sizes[1]),
        Layer(storage_size, layer_sizes[end], out_size),
    )
    state = [zeros(1, layer_sizes[1]),
             zeros(1, out_size)]
[is, state]
end

(is::IS)(states, in_1, in_2) =
begin
    out_1, state_1 = is.layer_1(states[1], in_1, in_2)
    out, state_out = is.layer_out(states[end], in_1, out_1)
[out, [state_1, state_out]]
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
    is, is_state = IS(storage_size, hm_vectors, vector_size, is_layers, storage_size)
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
    is, is_state = IS(storage_size, hm_vectors, vector_size, is_layers, storage_size)
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





attend(dec_storage, enc_storages, enc_vectors) =
begin
    similarities = soft([sum(dec_storage .* enc_storage) for enc_storage in enc_storages])
    # similarities = [sum(dec_storage .* enc_storage) for enc_storage in enc_storages]
# enc_vectors[argmax(reshape(similarities, length(similarities)))]
sum([vec .* sim for (vec, sim) in zip(enc_vectors,similarities)])
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
    storage_size = length(enc_state[2][end])
    enc_outs = [zeros(1, storage_size), nothing]
    enc_gs_time, enc_go_time = [], []
    for timestep in x
        enc_outs, enc_state = enc(enc_state, enc_outs[1], timestep)
        push!(enc_gs_time, enc_outs[1])
        push!(enc_go_time, enc_outs[2])
    end

    # transfer_states!(enc_state, dec_state)

    dec_outs = enc_outs
    # dec_outs = [zeros(1, storage_size), [zeros(1, vector_size) for i in 1:hm_vectors]]
    if dec_state == nothing dec_state = enc_state end # [zeros(1, storage_size), [zeros(1, vector_size) for i in 1:hm_vectors]]
    dec_go_time = []
    for timestep in 1:length_y
        dec_outs, dec_state = dec(dec_state, dec_outs[1], dec_outs[2], enc_gs_time, enc_go_time)
        push!(dec_go_time, dec_outs[2])
    end
dec_go_time
end


# using Statistics: norm

grads(result, encoder, decoder) =
begin
    grads = []
    for model in [encoder, decoder]
        for mfield in fieldnames(typeof(model))
            net = getfield(model, mfield)
            for nfield in fieldnames(typeof(net))
                layer = getfield(net, nfield)
                for lfield in fieldnames(typeof(layer))
                    gradient = grad(result, getfield(layer, lfield))
                    push!(grads, gradient)
                    # @show norm(gradient)
                end
            end
        end
    end
grads
end

upd!(encoder, decoder, grads, lr) =
begin
    i = 0
    for model in [encoder, decoder]
        for mfield in fieldnames(typeof(model))
            net = getfield(model, mfield)
            for nfield in fieldnames(typeof(net))
                layer = getfield(net, nfield)
                for lfield in fieldnames(typeof(layer))
                    i +=1
                    g = grads[i]
                    if g != nothing # TODO : rm dis check
                        setfield!(layer, lfield, Param(getfield(layer, lfield) - g.*lr))
                    end
                end
            end
        end
    end
end

upd_rms!(encoder, decoder, grads, lr, accu_grads; alpha=0.9) =
begin
    i = 0
    for model in [encoder, decoder]
        for mfield in fieldnames(typeof(model))
            net = getfield(model, mfield)
            for nfield in fieldnames(typeof(net))
                layer = getfield(net, nfield)
                for lfield in fieldnames(typeof(layer))
                    i +=1
                    accu_grads[i] = alpha .* accu_grads[i] .+ (1 - alpha) .* grads[i].^2
                    setfield!(layer, lfield, Param(getfield(layer, lfield) - grads[i].*lr./sqrt.(accu_grads[i].+1e-8)))
                end
            end
        end
    end
end


soft(arr) = ((arr) -> arr/sum(arr))(exp.(arr))

sqe(out, y) = sum(out .- y) ^2

sequence_loss(out, y) =
begin
    sum([sum(sum([sqe(out_e, y_e) for (out_e,y_e) in zip(out_t,y_t)])) for (out_t,y_t) in zip(out, y)])
end


drop(x; rate=0.1) = rand() < rate ? x : 0.0
begin
    # for i in 1:length(array):
    #     if rand() < rate
    #         array[i]
#     new_arr = [rand() < rate ? 0.0 : e for e in array]
# reshape(new_arr, size(array))
# array
end
