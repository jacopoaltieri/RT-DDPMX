import torch
import utils

def analyze_sparsity(model):
    """
    Analyze the sparsity of each layer in the model and identify pruned layers.
    """
    print("Analyzing model sparsity...")
    pruned_layers = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_zeros = torch.sum(param.data == 0).item()
            total_params = param.numel()
            sparsity = num_zeros / total_params * 100
            print(f"Layer: {name} | Total Params: {total_params} | Zeros: {num_zeros} | Sparsity: {sparsity:.2f}%")
            
            # Identify fully pruned layers
            if sparsity == 100.0:
                pruned_layers.append(name)
    return pruned_layers

def slim_model(model, pruned_layers):
    """
    Creates a slimmed-down version of the model by removing fully pruned layers.
    """
    print("\nSlimming down the model by removing fully pruned layers...")
    for layer_name in pruned_layers:
        layer_name_parts = layer_name.split('.')
        module = model
        for part in layer_name_parts[:-1]:
            module = getattr(module, part)
        delattr(module, layer_name_parts[-1])  # Remove the pruned layer
        print(f"Removed layer: {layer_name}")
    return model

def visualize_pruned_filters(model):
    """
    Visualize the number of pruned filters or channels in convolutional layers.
    """
    print("\nVisualizing pruned filters in convolutional layers...")
    for name, param in model.named_parameters():
        if "conv" in name and len(param.shape) == 4:  # Convolutional layers
            num_pruned_filters = torch.sum(torch.all(param.data == 0, dim=(1, 2, 3))).item()
            total_filters = param.shape[0]
            print(f"Layer: {name} | Total Filters: {total_filters} | Pruned Filters: {num_pruned_filters}")

def save_model(model, output_path, save_as_onnx=False, onnx_path="slimmed_model.onnx", input_shape=(1, 6, 256, 256), max_timesteps=100, device="cuda"):
    """
    Saves the model to disk in .pt or .onnx format.

    Parameters:
    - model: The model to save.
    - output_path: Path to save the PyTorch model (.pt).
    - save_as_onnx: If True, saves the model in ONNX format.
    - onnx_path: Path to save the ONNX model.
    - input_shape: The shape of the image input (batch_size, channels, height, width).
    - max_timesteps: The maximum value for the timestep input.
    - device: The device to use for saving and exporting.
    """
    print(f"\nSaving slimmed model to '{output_path}'")
    torch.save(model.state_dict(), output_path)
    
    if save_as_onnx:
        print(f"Saving slimmed model as ONNX to '{onnx_path}'")
        
        # Move the model to the specified device
        model.to(device)

        # Create dummy inputs on the same device as the model
        dummy_image = torch.randn(*input_shape, device=device)  # Random tensor for the image
        dummy_timestep = torch.randint(0, max_timesteps, (input_shape[0],), dtype=torch.long, device=device)  # Random timestep tensor

        # Export the model with the correct inputs
        torch.onnx.export(
            model,
            (dummy_image, dummy_timestep),  # Tuple of inputs
            onnx_path,
            opset_version=11,
            input_names=["image", "timestep"],
            output_names=["output"]
        )
        print("ONNX model saved successfully.")



def main():
    # Specify paths and configurations
    config_path = "cfg_256.yaml"  # Path to the YAML configuration file
    model_path = "unet_snapshot.pt"  # Path to the pruned model file
    output_path = "slimmed_model.pt"  # Path to save the slimmed model
    save_as_onnx = True  # Set to True to save as ONNX
    onnx_path = "slimmed_model.onnx"  # Path for the ONNX model
    input_shape = (6, 1, 256, 256)  # Input shape for ONNX export
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load configuration
    config = utils.load_yaml(config_path)

    # Load the model using the provided load_model function
    model = utils.load_model(config, model_path, device)

    # Step 1: Analyze sparsity and identify fully pruned layers
    pruned_layers = analyze_sparsity(model)
    
    # Step 2: Slim down the model by removing fully pruned layers
    slimmed_model = slim_model(model, pruned_layers)
    
    # Step 3: Visualize pruned filters in convolutional layers
    visualize_pruned_filters(slimmed_model)
    
    # Step 4: Save the slimmed model in .pt or .onnx format
    save_model(slimmed_model, output_path, save_as_onnx=save_as_onnx, onnx_path=onnx_path, input_shape=input_shape, max_timesteps=config["model"]["max_timesteps"],device=device)

if __name__ == "__main__":
    main()
