using Knet: @diff, Param, value, grad
using Knet: sigm, tanh, softmax


input_size  = 42
hiddens     = [86]
output_size = 52


hm_data = 50
seq_len = 100

hm_epochs = 10
lr        = .001


mk_model(in,hiddens,out,type) =
begin
    model = []
    hm_layers = length(hiddens)+1
    for i in 1:hm_layers
        if i == 1
            layer = type(in, hiddens[1])
        elseif i == hm_layers
            layer = type(hiddens[end],out)
        else
            layer = type(hiddens[i-1],hiddens[i])
        end
        push!(model, layer)
    end
model
end


prop_model(model, seq) =
begin
    out = seq[1]
    outs = []
    for t in 1:length(seq)+1
        for layer in model
            out = layer(out) # if force_teach layer(seq[i+1])
        end
        push!(outs, out)
    end
    deleteat!(outs, 1)
    for layer in model
        setfield!(layer, :state, zeros(1,size(getfield(layer, :wfs))[end]))
    end
outs
end


mutable struct GRU
    wfi::Param
    wfs::Param
    wii::Param
    wki::Param
    wks::Param
    state
end

GRU(in_size, layer_size) =
begin
    sq = sqrt(2/(in_size+layer_size))

    wfi = Param(2*sq .* randn(in_size,layer_size)    .-sq)
    wfs = Param(2*sq .* randn(layer_size,layer_size) .-sq)

    wki = Param(2*sq .* randn(in_size,layer_size)    .-sq)
    wks = Param(2*sq .* randn(layer_size,layer_size) .-sq)

    wii = Param(2*sq .* randn(in_size,layer_size)    .-sq)

    state = zeros(1,layer_size)
    layer = GRU(wfi, wfs, wii, wki, wks, state)
layer
end

(layer::GRU)(in) =
begin
    focus  = sigm.(in * layer.wfi + layer.state * layer.wfs)
    keep   = sigm.(in * layer.wki + layer.state * layer.wks)
    interm = tanh.(in * layer.wii + layer.state .* focus)

    layer.state = keep .* layer.state + (1 .- keep) .* interm
layer.state
end



mutable struct LSTM
    wki::Param
    wks::Param
    wfi::Param
    wfs::Param
    wsi::Param
    wss::Param
    wii::Param
    wis::Param
    state
end

LSTM(in_size, layer_size) =
begin
    sq = sqrt(2/(in_size+layer_size))

    wki = Param(2*sq .* randn(in_size,layer_size)    .-sq)
    wks = Param(2*sq .* randn(layer_size,layer_size) .-sq)

    wfi = Param(2*sq .* randn(in_size,layer_size)    .-sq)
    wfs = Param(2*sq .* randn(layer_size,layer_size) .-sq)

    wsi = Param(2*sq .* randn(in_size,layer_size)    .-sq)
    wss = Param(2*sq .* randn(layer_size,layer_size) .-sq)

    wii = Param(2*sq .* randn(in_size,layer_size)    .-sq)
    wis = Param(2*sq .* randn(in_size,layer_size)   .-sq)

    state = zeros(1,layer_size)
    layer = LSTM(wki, wks, wfi, wfs, wsi, wss, wii, wis, state)
layer
end

(layer::LSTM)(in) =
begin
    keep   = sigm.(in * layer.wki + layer.state * layer.wfs)
    forget = sigm.(in * layer.wfi + layer.state * layer.wks)
    show   = sigm.(in * layer.wsi + layer.state * layer.wks)
    interm = tanh.(in * layer.wii + layer.state * layer.wis)
    layer.state = forget .* layer.state + keep .* interm
    out = show .* layer.state
out
end



model_type  = GRU

main() =
begin
    model = mk_model(input_size,hiddens,output_size,model_type)

    data = [[randn(1,input_size) for _ in 1:seq_len] for __ in hm_data]

    for e in 1:hm_epochs

        loss = 0

        for seq in data

            d = @diff begin

                input  = seq[1:end-1]
                label  = seq[2:end]
                output = prop_model(model,input)
                sum(sum([(lbl-out).^2 for (lbl,out) in zip(label,output)]))
            end

            loss += value(d)

            for layer in model
                for param in fieldnames(model_type)
                    g = grad(d,getfield(layer, param))
                    if g != nothing
                        setfield!(layer, param, Param(getfield(layer, param)-lr.*g))
                    end
                end
            end

        end
    println("epoch $e loss: $loss")
    end
end
main()
