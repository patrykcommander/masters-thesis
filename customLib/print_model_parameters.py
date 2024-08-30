from torch import nn
def print_model_parameters(model):
    total_params = 0
    for name, layer in model.named_modules():
        if name == "":
            continue

        if isinstance(layer, nn.Module) and len(list(layer.parameters())) > 0:
            layer_params = sum(param.numel() for param in layer.parameters() if param.requires_grad)
            statement = f"Layer {name}:" if name != "" else "Total params:"
            print(f"{statement} {layer_params} parameters")
            total_params += layer_params

    print(f"Total params: {total_params}")
    