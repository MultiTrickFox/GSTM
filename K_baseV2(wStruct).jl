using Knet: Param, @diff, value, grad

import Statistics.mean



mutable struct Layer
    w::Param
    b::Param
end

function Layer(in_size::Int, out_size::Int)
    w = Param(randn(in_size,out_size)/sqrt(out_size))
    b = Param(ones(1,out_size))
    Layer(w,b)
end

function (layer::Layer)(input)
    input * layer.w + layer.b
end


mutable struct Model
    layers::Array{Layer}
end

function Model(layers)
    Model(layers)
end

function (model::Model)(input)
    for layer in model.layers
        input = layer(input)
    end
input
end


grads(results, model::Model) =
begin
  layer_fields = fieldnames(Layer)
  grads = []
  for layer in model.layers
    for param in layer_fields
      push!(grads, grad(results, getfield(layer, param)))
    end
  end
grads
end





in_size       = 12
hidden_sizes  = [9, 7, 5]
out_size      = 3



hm_layers = length(hidden_sizes)+1
model_internal = Layer[]

for i in 1:hm_layers
    if     i == 1         prev_size = in_size           ; next_size = hidden_sizes[i]
    elseif i == hm_layers prev_size = hidden_sizes[end] ; next_size = out_size
    else                  prev_size = hidden_sizes[i-1] ; next_size = hidden_sizes[i]
    end
    push!(model_internal, Layer(prev_size, next_size))
end

model = Model(model_internal)



hm_data = 1
dataset = []

for i in 1:hm_data
  x = randn(1, in_size)
  y = randn(1, out_size)
  push!(dataset, [x,y])
end



# println(dataset[1][2])
# println(model(dataset[1][1]))



mse(output, label) =
begin
  loss = mean((output - label) .^2)
loss
end

sgd!(model, grads, lr) =
begin
  layer_fields = fieldnames(Layer)
  i = 0
  for layer in model.layers
    for param in layer_fields
      i +=1
      setfield!(layer, param, Param(getfield(layer, param) - grads[i] * lr))
    end
  end
end

train(model, dataset, hm_epochs, lr; criterion=mse, optimizer=sgd!) =
begin
  for ep in 1:hm_epochs
    loss = 0
    for (x,y) in dataset
      results = @diff mse(model(x), y)
      loss += value(results)
      sgd!(model, grads(results, model), lr)
    end
    println("Epoch ", ep, " Loss ", loss)
  end
end





@time train(model, dataset, 100, .01)
