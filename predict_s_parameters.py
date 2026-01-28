#!/usr/bin/env python3
"""
Use trained RINN model to predict S-parameters for fixed geometric parameters at different frequencies
and plot comparison between actual and predicted values
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
import sys

# Add project path
sys.path.append(str(Path(__file__).parent))

from R_INN_model.rinn_model import RINNModel


def load_model_and_config(checkpoint_dir):
    """Load trained model and configuration"""
    checkpoint_path = Path(checkpoint_dir)
    
    # Load configuration file
    config_path = checkpoint_path / "training_config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Extract model configuration
    model_config = config['model_config']
    data_info = config['data_info']
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RINNModel(
        input_dim=model_config['input_dim'],
        hidden_dim=model_config['hidden_dim'],
        num_blocks=model_config['num_blocks'],
        num_stages=model_config['num_stages'],
        num_cycles_per_stage=model_config['num_cycles_per_stage'],
        ratio_toZ_after_flowstage=model_config['ratio_toZ_after_flowstage'],
        ratio_x1_x2_inAffine=model_config['ratio_x1_x2_inAffine']
    ).to(device)
    
    # Load model weights
    model_path = checkpoint_path / "best_model.pth"
    if model_path.exists():
        checkpoint = torch.load(model_path, map_location=device)
        # Extract model_state_dict from checkpoint dictionary
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model weights: {model_path}")
            print(f"Trained for {checkpoint.get('epoch', 'unknown')+1} epochs")
            print(f"Validation loss: {checkpoint.get('val_loss', 'unknown')}")
        else:
            model.load_state_dict(checkpoint)
            print(f"Loaded model weights: {model_path}")
    else:
        print(f"Warning: Model weights file not found {model_path}")
    
    model.eval()
    
    return model, config, device


def clip_y_data(y, y_lower_bound, y_upper_bound):
    """Clip Y data to handle outliers (same as training script)"""
    return np.clip(y, y_lower_bound, y_upper_bound)


def normalize_data(data, median, iqr):
    """Robust normalization using median and IQR"""
    return (data - np.array(median)) / (np.array(iqr) + 1e-8)


def denormalize_data(data, median, iqr):
    """Robust denormalization using median and IQR"""
    return data * np.array(iqr) + np.array(median)


def find_matching_column(csv_data, h1, h2, h3, h_c1, h_c2):
    """Find column names in CSV data matching geometric parameters"""
    # Build target column name pattern
    target_pattern = f"H1='{h1}mm' H2='{h2}mm' H3='{h3}mm' H_C1='{h_c1}mm' H_C2='{h_c2}mm'"
    
    # Find matching columns
    matching_cols = [col for col in csv_data.columns if target_pattern in col]
    
    if not matching_cols:
        print(f"Warning: No matching geometric parameters found: {target_pattern}")
        return None
    
    # Return first matching column (real and imaginary parts)
    return matching_cols[0], matching_cols[0].replace("re(S(1,1))", "im(S(1,1))")


def predict_s_parameters(model, device, config, h1, h2, h3, h_c1, h_c2, csv_path):
    """Predict S-parameters for fixed geometric parameters at different frequencies"""
    
    # Load CSV data
    csv_data = pd.read_csv(csv_path)
    
    # Extract frequency column
    frequencies = csv_data['Freq [GHz]'].values
    
    # Find matching columns
    re_col, im_col = find_matching_column(csv_data, h1, h2, h3, h_c1, h_c2)
    
    if re_col is None:
        return None, None, None
    
    # Extract actual values
    actual_re = csv_data[re_col].values
    actual_im = csv_data[im_col].values
    
    # Prepare input data
    data_info = config['data_info']
    x_median = np.array(data_info['x_mean'])  # Actually median in robust normalization
    x_iqr = np.array(data_info['x_std'])      # Actually IQR in robust normalization
    y_median = np.array(data_info['y_mean'])  # Actually median in robust normalization
    y_iqr = np.array(data_info['y_std'])      # Actually IQR in robust normalization
    
    # Calculate Y clipping bounds (same as training script)
    # Q1 = median - IQR/2, Q3 = median + IQR/2 (approximation for robust normalization)
    y_q1 = y_median - y_iqr / 2
    y_q3 = y_median + y_iqr / 2
    y_lower_bound = y_q1 - 3 * y_iqr
    y_upper_bound = y_q3 + 3 * y_iqr
    
    # Clip actual Y data (same as training script)
    actual_re_clipped = clip_y_data(actual_re, y_lower_bound[0], y_upper_bound[0])
    actual_im_clipped = clip_y_data(actual_im, y_lower_bound[1], y_upper_bound[1])
    
    # Fixed geometric parameters
    geo_params = np.array([h1, h2, h3, h_c1, h_c2])
    
    # Predict for each frequency
    predicted_re_list = []
    predicted_im_list = []
    
    for freq in frequencies:
        # Build input: [h1, h2, h3, h_c1, h_c2, freq]
        x = np.array([h1, h2, h3, h_c1, h_c2, freq])
        
        # Robust normalization
        x_normalized = normalize_data(x, x_median, x_iqr)
        
        # Build left input: X + zero padding (2 dimensions)
        padding_dim = 2  # Y dimension
        left_input = np.concatenate([x_normalized, np.zeros(padding_dim)])
        
        # Convert to tensor
        x_tensor = torch.FloatTensor(left_input).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            # Use model forward pass
            predicted_right, _, _ = model(x_tensor, return_intermediate=True)
            
            # Extract predicted Y (first 2 dimensions)
            predicted_y = predicted_right[:, :2]
            
            # Robust denormalization
            predicted_y_denorm = denormalize_data(
                predicted_y.cpu().numpy()[0],
                y_median,
                y_iqr
            )
            
            predicted_re_list.append(predicted_y_denorm[0])
            predicted_im_list.append(predicted_y_denorm[1])
    
    # Clip predicted values to same bounds
    predicted_re_clipped = clip_y_data(np.array(predicted_re_list), y_lower_bound[0], y_upper_bound[0])
    predicted_im_clipped = clip_y_data(np.array(predicted_im_list), y_lower_bound[1], y_upper_bound[1])
    
    return frequencies, (actual_re_clipped, actual_im_clipped), (predicted_re_clipped, predicted_im_clipped)


def plot_comparison(frequencies, actual, predicted, geo_params, save_path):
    """Plot comparison between actual and predicted values"""
    
    actual_re, actual_im = actual
    predicted_re, predicted_im = predicted
    
    h1, h2, h3, h_c1, h_c2 = geo_params
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Real part comparison
    axes[0].plot(frequencies, actual_re, 'b-', linewidth=2, label='Actual', marker='o', markersize=4)
    axes[0].plot(frequencies, predicted_re, 'r--', linewidth=2, label='Predicted', marker='s', markersize=4)
    axes[0].set_xlabel('Frequency (GHz)', fontsize=12)
    axes[0].set_ylabel('S11 Real Part', fontsize=12)
    axes[0].set_title(f'S11 Real Part Comparison\nGeometric Parameters: H1={h1}mm, H2={h2}mm, H3={h3}mm, H_C1={h_c1}mm, H_C2={h_c2}mm', fontsize=14)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Imaginary part comparison
    axes[1].plot(frequencies, actual_im, 'b-', linewidth=2, label='Actual', marker='o', markersize=4)
    axes[1].plot(frequencies, predicted_im, 'r--', linewidth=2, label='Predicted', marker='s', markersize=4)
    axes[1].set_xlabel('Frequency (GHz)', fontsize=12)
    axes[1].set_ylabel('S11 Imaginary Part', fontsize=12)
    axes[1].set_title(f'S11 Imaginary Part Comparison\nGeometric Parameters: H1={h1}mm, H2={h2}mm, H3={h3}mm, H_C1={h_c1}mm, H_C2={h_c2}mm', fontsize=14)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to: {save_path}")
    
    return fig


def calculate_metrics(actual, predicted):
    """Calculate evaluation metrics"""
    actual_re, actual_im = actual
    predicted_re, predicted_im = predicted
    
    # MSE
    mse_re = np.mean((actual_re - predicted_re) ** 2)
    mse_im = np.mean((actual_im - predicted_im) ** 2)
    
    # MAE
    mae_re = np.mean(np.abs(actual_re - predicted_re))
    mae_im = np.mean(np.abs(actual_im - predicted_im))
    
    # R² score
    ss_res_re = np.sum((actual_re - predicted_re) ** 2)
    ss_tot_re = np.sum((actual_re - np.mean(actual_re)) ** 2)
    r2_re = 1 - (ss_res_re / ss_tot_re) if ss_tot_re != 0 else 0
    
    ss_res_im = np.sum((actual_im - predicted_im) ** 2)
    ss_tot_im = np.sum((actual_im - np.mean(actual_im)) ** 2)
    r2_im = 1 - (ss_res_im / ss_tot_im) if ss_tot_im != 0 else 0
    
    metrics = {
        'MSE': {'Real': mse_re, 'Imaginary': mse_im},
        'MAE': {'Real': mae_re, 'Imaginary': mae_im},
        'R2': {'Real': r2_re, 'Imaginary': r2_im}
    }
    
    return metrics


def main():
    """Main function to run S-parameter prediction"""
    
    # Model checkpoint directory
    checkpoint_dir = "model_checkpoints_rinn/rinn_sparameter_20260126_231618"
    
    # Load model and configuration
    print("Loading model and configuration...")
    model, config, device = load_model_and_config(checkpoint_dir)
    
    # Print configuration
    print("\n" + "="*60)
    print("Model Configuration:")
    print("="*60)
    print(f"Input dimension: {config['model_config']['input_dim']}")
    print(f"Hidden dimension: {config['model_config']['hidden_dim']}")
    print(f"Number of blocks: {config['model_config']['num_blocks']}")
    print(f"Number of stages: {config['model_config']['num_stages']}")
    print(f"Ratio to Z after flow stage: {config['model_config']['ratio_toZ_after_flowstage']}")
    print(f"Ratio x1:x2 in affine: {config['model_config']['ratio_x1_x2_inAffine']}")
    
    print("\n" + "="*60)
    print("Data Information:")
    print("="*60)
    data_info = config['data_info']
    x_median = np.array(data_info['x_mean'])
    x_iqr = np.array(data_info['x_std'])
    y_median = np.array(data_info['y_mean'])
    y_iqr = np.array(data_info['y_std'])
    print(f"X median (mean): {x_median}")
    print(f"X IQR (std): {x_iqr}")
    print(f"Y median (mean): {y_median}")
    print(f"Y IQR (std): {y_iqr}")
    
    # Calculate clipping bounds for debug
    y_q1 = y_median - y_iqr / 2
    y_q3 = y_median + y_iqr / 2
    y_lower_bound = y_q1 - 3 * y_iqr
    y_upper_bound = y_q3 + 3 * y_iqr
    print(f"\nY clipping bounds:")
    print(f"  Re part: [{y_lower_bound[0]:.4f}, {y_upper_bound[0]:.4f}]")
    print(f"  Im part: [{y_lower_bound[1]:.4f}, {y_upper_bound[1]:.4f}]")
    
    # CSV data path
    csv_path = "data/S Parameter Plot 7.csv"
    
    # Test case 1: First geometric parameter combination from CSV data
    print("\n" + "="*60)
    print("Test Case 1: h1=3.11117, h2=3.65863, h3=2.9958, h_c1=3.35774, h_c2=3.01181")
    print("="*60)
    
    frequencies, actual, predicted = predict_s_parameters(
        model, device, config,
        h1=3.11117, h2=3.65863, h3=2.9958, h_c1=3.35774, h_c2=3.01181,
        csv_path=csv_path
    )
    
    if frequencies is not None:
        # Calculate error metrics
        actual_re, actual_im = actual
        predicted_re, predicted_im = predicted
        
        # Calculate errors
        error_re = np.abs(actual_re - predicted_re)
        error_im = np.abs(actual_im - predicted_im)
        
        print(f"\nError Statistics:")
        print(f"  Real part - Mean: {np.mean(error_re):.6f}, Max: {np.max(error_re):.6f}")
        print(f"  Imag part - Mean: {np.mean(error_im):.6f}, Max: {np.max(error_im):.6f}")
        
        print(f"\nValue Ranges:")
        print(f"  Actual Re: [{np.min(actual_re):.4f}, {np.max(actual_re):.4f}]")
        print(f"  Predicted Re: [{np.min(predicted_re):.4f}, {np.max(predicted_re):.4f}]")
        print(f"  Actual Im: [{np.min(actual_im):.4f}, {np.max(actual_im):.4f}]")
        print(f"  Predicted Im: [{np.min(predicted_im):.4f}, {np.max(predicted_im):.4f}]")
        
        # Plot results
        plot_comparison(frequencies, actual, predicted, 
                         (3.11117, 3.65863, 2.9958, 3.35774, 3.01181),
                         "results/s_parameter_prediction_test1.png")
    else:
        print("No matching data found for test case 1")
    
    # Test case 2: Different geometric parameters from CSV data
    print("\n" + "="*60)
    print("Test Case 2: h1=3.30562, h2=3.96351, h3=3.12063, h_c1=3.35774, h_c2=3.04255")
    print("="*60)
    
    frequencies2, actual2, predicted2 = predict_s_parameters(
        model, device, config,
        h1=3.30562, h2=3.96351, h3=3.12063, h_c1=3.35774, h_c2=3.04255,
        csv_path=csv_path
    )
    
    if frequencies2 is not None:
        # Calculate error metrics
        actual_re2, actual_im2 = actual2
        predicted_re2, predicted_im2 = predicted2
        
        # Calculate errors
        error_re2 = np.abs(actual_re2 - predicted_re2)
        error_im2 = np.abs(actual_im2 - predicted_im2)
        
        print(f"\nError Statistics:")
        print(f"  Real part - Mean: {np.mean(error_re2):.6f}, Max: {np.max(error_re2):.6f}")
        print(f"  Imag part - Mean: {np.mean(error_im2):.6f}, Max: {np.max(error_im2):.6f}")
        
        print(f"\nValue Ranges:")
        print(f"  Actual Re: [{np.min(actual_re2):.4f}, {np.max(actual_re2):.4f}]")
        print(f"  Predicted Re: [{np.min(predicted_re2):.4f}, {np.max(predicted_re2):.4f}]")
        print(f"  Actual Im: [{np.min(actual_im2):.4f}, {np.max(actual_im2):.4f}]")
        print(f"  Predicted Im: [{np.min(predicted_im2):.4f}, {np.max(predicted_im2):.4f}]")
        
        # Plot results
        plot_comparison(frequencies2, actual2, predicted2, 
                         (3.30562, 3.96351, 3.12063, 3.35774, 3.04255),
                         "results/s_parameter_prediction_test2.png")
    else:
        print("No matching data found for test case 2")
    
    print("\n" + "="*60)
    print("Prediction complete!")
    print("="*60)


if __name__ == "__main__":
    main()