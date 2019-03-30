using Distributed: @everywhere, @distributed, addprocs, procs
if length(procs()) == 1 addprocs(3) end

@everywhere using Knet: @diff, Param, value, grad
@everywhere using Knet: sigm, tanh, softmax
# using Gadfly: plot
# using Statistics: norm



model_types = ["GRU","GSTM","LSTM"]

input_size  = 2
hiddens     = []
output_size = 1


hm_data    = 1_000
batch_size = 50
hm_test    = 200

seq_len = 200
# 100 # 200 # 500 # 1_000 # 2_000


hm_epochs = 50
lr        = .001


alpha_moments = 0.9
alpha_accugrads = 0.999


layer_test = []#[10], [12], [16], [20], [25], [28], [30], [32], [36], [40], [45], [48], [50], [52], [56], [58], [60], [62], [64], [68], [72], [74], [80], [84], [88], [92], [96], [100]]
hm_trials  = 1

verbose = true


mk_model(in,hiddens,out,type) =
begin
    model = []
    hm_layers = length(hiddens)+1
    if hm_layers != 1
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
    else
        push!(model, type(in,out))
    end
model
end


@everywhere prop_model(model, seq) =
begin
    outs = []
    for t in seq
        out = t
        for layer in model
            out = layer(out)
        end
        push!(outs, out)
    end
    for layer in model
        setfield!(layer, :state, zeros(1,size(getfield(layer, :wfi))[end]))
    end
outs
end



@everywhere mutable struct GRU
    wfi::Param
    wfs::Param
    wii::Param
    wki::Param
    wks::Param
    state
end

@everywhere GRU(in_size, layer_size) =
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

@everywhere (layer::GRU)(in) =
begin
    focus  = sigm.(in * layer.wfi + layer.state * layer.wfs)
    keep   = sigm.(in * layer.wki + layer.state * layer.wks)
    interm = tanh.(in * layer.wii + layer.state .* focus)

    layer.state = keep .* layer.state + (1 .- keep) .* interm
layer.state
end



@everywhere mutable struct LSTM
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

@everywhere LSTM(in_size, layer_size) =
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

@everywhere (layer::LSTM)(in) =
begin
    keep   = sigm.(in * layer.wki + layer.state * layer.wfs)
    forget = sigm.(in * layer.wfi + layer.state * layer.wks)
    show   = sigm.(in * layer.wsi + layer.state * layer.wks)
    interm = tanh.(in * layer.wii + layer.state * layer.wis)
    layer.state = forget .* layer.state + keep .* interm
    out = show .* tanh.(layer.state)
out
end



@everywhere mutable struct GSTM
    wfi::Param
    wfs::Param
    wri::Param
    wki::Param
    wks::Param
    wsi::Param
    wss::Param
    state
end

@everywhere GSTM(in_size, layer_size) =
begin
    sq = sqrt(2/(in_size+layer_size))

    wfi = Param(2*sq .* randn(in_size,layer_size)    .-sq)
    wfs = Param(2*sq .* randn(layer_size,layer_size) .-sq)

    wri = Param(2*sq .* randn(in_size,layer_size)    .-sq)

    wki = Param(2*sq .* randn(in_size,layer_size)    .-sq)
    wks = Param(2*sq .* randn(layer_size,layer_size) .-sq)

    wsi = Param(2*sq .* randn(in_size,layer_size)    .-sq)
    wss = Param(2*sq .* randn(layer_size,layer_size) .-sq)

    state = zeros(1,layer_size)
    layer = GSTM(wfi, wfs, wri, wki, wks, wsi, wss, state)
layer
end

@everywhere (layer::GSTM)(in) =
begin
    st = tanh.(layer.state)
    focus    = sigm.(in * layer.wfi + st * layer.wfs)
    reaction = tanh.(in * layer.wri + st .* focus)
    keep     = sigm.(in * layer.wki + st * layer.wks)
    show     = sigm.(in * layer.wsi + st * layer.wss)

    layer.state += reaction .* keep
    out = reaction .* show
out
end



grad_clip(x) =
if x < -1
    -1.0
elseif x > 1
    1.0
else
    x
end




main(model_name, data; hiddens=hiddens) =
begin
    model_type = (@eval $(Symbol(model_name)))

    model = mk_model(input_size,hiddens,output_size,model_type)

    moments = [zeros(size(getfield(layer,param))) for layer in model for param in fieldnames(typeof(layer))]
    accugrads = [zeros(size(getfield(layer,param))) for layer in model for param in fieldnames(typeof(layer))]

    losses = []

    test_sc1 = []
    test_sc2 = []


    for e in 1:hm_epochs

        loss = 0

        for batch in batchify(data, batch_size)

            results = @distributed (vcat) for seq in batch

                d = @diff begin

                    # input  = seq[1:end-1]
                    # label  = seq[2:end]

                    input = seq[1]
                    label = seq[2]

                    output = prop_model(model,input)[end]
                    sum(sum([(lbl-out).^2 for (lbl,out) in zip(label,output)]))

                end

                grads = []

                i = 0
                for layer in model
                    for param in fieldnames(model_type)
                        i +=1
                        g = grad(d,getfield(layer, param))
                        push!(grads, g)
                    end
                end

                grads, value(d)

            end

            for (grads, l) in results

                loss += l

                i = 0
                for layer in model
                    for param in fieldnames(model_type)
                        i +=1
                        g = grads[i]
                        if g != nothing

                            g = grad_clip.(g)

                            moments[i] = alpha_moments * moments[i] + (1 - alpha_moments) * g
                            accugrads[i] = alpha_accugrads * accugrads[i] + (1 - alpha_accugrads) * g .^2

                            moment_hat = moments[i] / (1 - alpha_moments .^ e)
                            accugrad_hat = accugrads[i] / (1 - alpha_accugrads .^ e)

                            setfield!(layer, param, Param(getfield(layer, param)-lr.*moment_hat/sqrt(sum(accugrad_hat) + 1e-8)))

                            # setfield!(layer, param, Param(getfield(layer, param)-lr.*g))
                        end

                    end
                end

            end

        end

    push!(losses, loss)
    verbose ? println("epoch $e loss: $loss") : ()

    # for (inp,lbl) in test_set
    #     println("model prediction: $(sum(prop_model(model,inp)[end]))")
    #     println("actual result: $lbl")
    # end

    test_score = sum([abs(lbl - sum(prop_model(model,inp)[end])) < .01 ? 1 : 0 for (inp,lbl) in test_set])
    test_score2 = sum([abs(lbl - sum(prop_model(model,inp)[end])) < .1 ? 1 : 0 for (inp,lbl) in test_set])

    push!(test_sc1, test_score)
    push!(test_sc2, test_score2)

    if e%1 == 0
        println("\t> test lo_acc: $(test_score2)")
        println("\t> test hi_acc: $(test_score)")
    end



    end


    # verbose ? (begin prev_loss = 999_999_999
    # for (e,loss) in enumerate(losses)
    #     if loss > prev_loss
    #         println("bad loss: ep $e \n \t $prev_loss to $loss")
    #     end
    #     prev_loss = loss
    # end end) : ()

    verbose ? println("\n\t\t $model_name summary:") : ()

    for loss in[losses[1], losses[trunc(Int,length(losses)*1/4)], losses[trunc(Int,length(losses)*2/4)], losses[trunc(Int,length(losses)*3/4)], losses[end]]
        println(loss)
    end ;

    # verbose ? println("progress : $((1-losses[end]/losses[1])*100)") : ()

    println(" ")


    # TODO : graph here.

    # test_sc1 is more accurate
    # test_sc2 is less accurate

    # plot(test_sc1)
    # plot(test_sc2)







(losses[1],losses[end])
end





# d = [[randn(1,input_size) for _ in 1:seq_len] for __ in hm_data]

d = []
for i in 1:hm_data
    y = 1
    x = []
    while ! (y < 1)
        x = []
        for t in 1:seq_len
            arr = [randn(), randn() < 0.2 ? 0 : 1]
            push!(x, reshape(arr, 1, length(arr)))
        end
        y = sum(e1*e2 for (e1,e2) in x)
    end
    push!(d, [x,y])
end

test_set = []
for i in 1:hm_test
    y = 1
    x = []
    while ! (y < 1)
        x = []
        for t in 1:seq_len
            arr = [randn(), randn() < 0.5 ? 0 : 1]
            push!(x, reshape(arr, 1, length(arr)))
        end
        y = sum(e1*e2 for (e1,e2) in x)
    end
    push!(test_set, [x,y])
end


batchify(data, bs) =
begin
    batches = []
    for b in 1:(trunc(Int,length(data)/bs))
        push!(batches, data[(b-1)*bs+1:b*bs])
    end
    hm_left = length(data)%bs
    if hm_left != 0
        push!(batches, data[end-(hm_left-1):end])
    end
batches
end





# runner of main. main of main

for model_type in model_types
    println("\n\t> Running: $model_type \n")
    if length(layer_test) > 0
        progresses = [[0.0,0.0] for _ in 1:length(layer_test)]
        for _ in 1:hm_trials
            data = [[randn(1,input_size) for _ in 1:seq_len] for __ in hm_data]
            for (i,l) in enumerate(layer_test)
                println("layers: $l")
                loss1, lossend = main(model_type, data, hiddens=l)
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
            println("$model_type $(hiddens[i]) : $(p[1]) to $(p[end]) : $((1-p[end]/p[1])*100)")
        end

    else main(model_type, d) end
end ; println("\n\ndone.\n")
