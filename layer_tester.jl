using Distributed: @everywhere, @distributed
using Knet: @diff, Param, value, grad
using Knet: sigm, tanh, softmax



model_types = ["GRU"]

input_size  = 42
hiddens     = [20]
output_size = 42


hm_data = 1_000
seq_len = 500

hm_epochs = 100
lr        = .001


layer_test = [[2], [5], [8], [10], [20], [22], [25], [28], [30], [32], [35], [40], [45], [50], [52], [55], [60], [64], [68]]#, [72], [74], [80]]
hm_trials  = 20


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
        setfield!(layer, :state, zeros(1,size(getfield(layer, :wfi))[end]))
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
    wis = Param(2*sq .* randn(layer_size,layer_size) .-sq)

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
    out = show .* tanh.(layer.state)
out
end



mutable struct GSTM
    wfi::Param
    wfs::Param
    wri::Param
    wk1i::Param
    wk1s::Param
    wk2i::Param
    wk2s::Param
    wsi::Param
    wss::Param
    state
end

GSTM(in_size, layer_size) =
begin
    sq = sqrt(2/(in_size+layer_size))

    wfi = Param(2*sq .* randn(in_size,layer_size)    .-sq)
    wfs = Param(2*sq .* randn(layer_size,layer_size) .-sq)

    wri = Param(2*sq .* randn(in_size,layer_size)    .-sq)

    wk1i = Param(2*sq .* randn(in_size,layer_size)    .-sq)
    wk1s = Param(2*sq .* randn(layer_size,layer_size) .-sq)

    wk2i = Param(2*sq .* randn(in_size,layer_size)    .-sq)
    wk2s = Param(2*sq .* randn(layer_size,layer_size) .-sq)

    wsi = Param(2*sq .* randn(in_size,layer_size)    .-sq)
    wss = Param(2*sq .* randn(layer_size,layer_size) .-sq)

    state = zeros(1,layer_size)
    layer = GSTM(wfi, wfs, wri, wk1i, wk1s, wk2i, wk2s, wsi, wss, state)
layer
end

(layer::GSTM)(in) =
begin
    focus    = sigm.(in * layer.wfi + layer.state * layer.wfs)
    reaction = tanh.(in * layer.wri + layer.state .* focus)
    keep1    = sigm.(in * layer.wk1i + layer.state * layer.wk1s)
    keep2    = sigm.(in * layer.wk2i + layer.state * layer.wk2s)
    show     = sigm.(in * layer.wsi + layer.state * layer.wss)

    layer.state = reaction .* keep1 + layer.state .* keep2
    out = reaction .* show # (show .* reaction) .* layer.state
out
end





verbose = false

main(model_name) =
begin
    model_type = (@eval $(Symbol(model_name)))

    model = mk_model(input_size,hiddens,output_size,model_type)

    data = [[randn(1,input_size) for _ in 1:seq_len] for __ in hm_data]

    losses = []

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

        end ; push!(losses, loss)
    verbose ? println("epoch $e loss: $loss") : ()
    end


    verbose ? (begin prev_loss = 999_999_999
    for (e,loss) in enumerate(losses)
        if loss > prev_loss
            println("bad loss: ep $e \n \t $prev_loss to $loss")
        end
        prev_loss = loss
    end end) : ()

    verbose ? println("\n\t\t $model_name summary:") : ()

    for loss in[losses[1], losses[trunc(Int,length(losses)*1/4)], losses[trunc(Int,length(losses)*2/4)], losses[trunc(Int,length(losses)*3/4)], losses[end]]
        println(loss)
    end ; println(" ")


    # TODO : graph here.

(losses[1],losses[end])
end



for model_type in model_types
    println("\n\t> Running: $model_type \n")
    if length(layer_test) > 0
        progresses = [[0.0,0.0] for _ in 1:length(layer_test)]
        for _ in 1:hm_trials
            for (i,l) in enumerate(layer_test)
                hiddens = l ; println("layers: $hiddens")
                loss1, lossend = main(model_type)
                progresses[i][1] += loss1
                progresses[i][end] += lossend
            end

            fittest = argmax([(1-progress[end]/progress[1])*100 for progress in progresses])
            println("** Current Optimal layer size: $(layer_test[fittest]) \n")

        end

        fittest = argmax([(1-progress[end]/progress[1])*100 for progress in progresses])
        println(">> General Optimal layer size: $(layer_test[fittest]) \n")

        println("\n Progress list: \n")
        for (i,p) in enumerate(progresses)
            println("$model_type $(hiddens[i]) $((1-p[end]/p[1])*100)")
            println("$p\n")
        end

    else main(model_type) end
end
