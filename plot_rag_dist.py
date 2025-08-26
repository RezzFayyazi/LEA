import matplotlib.pyplot as plt
import numpy as np
import json

def extract_rag_value(data):
    if data is None:
        return 0.0
    
    if isinstance(data, (int, float)):
        return float(data)
    
    if isinstance(data, dict):
        target_key = "xy=True,x?y=False" # this is the RAG key for the LEA results (as it transfers from Independent (True) to Dependent (False)
        if target_key in data and data[target_key] is not None:
            return float(data[target_key])
    
    return 0.0

def load_model_data(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    
    rag_values = []
    items = data.items() if isinstance(data, dict) else enumerate(data)
    
    for _, sample_data in items:
        percentages = None
        if isinstance(sample_data, dict):
            if "percentages" in sample_data:
                percentages = sample_data["percentages"]
            elif "results" in sample_data and "percentages" in sample_data["results"]:
                percentages = sample_data["results"]["percentages"]
            else:
                percentages = sample_data
        else:
            percentages = sample_data
        
        rag_val = extract_rag_value(percentages)
        rag_values.append(rag_val)
    
    rag_arr = np.array(rag_values, dtype=float)
    
    if rag_arr.size > 0 and np.max(rag_arr) <= 1.0:
        rag_arr = rag_arr * 100.0
    
    return np.clip(rag_arr, 0.0, 100.0).tolist()

def split_by_years(values, samples_per_year=50):
    values_array = np.array(values)
    yearly_data = {}
    
    n_complete_years = len(values_array) // samples_per_year
    
    for year_idx in range(n_complete_years):
        year = 2016 + year_idx
        start_idx = year_idx * samples_per_year
        end_idx = start_idx + samples_per_year
        yearly_data[year] = values_array[start_idx:end_idx]
    
    remaining_samples = len(values_array) % samples_per_year
    if remaining_samples > 0 and n_complete_years < 10:
        year = 2016 + n_complete_years
        start_idx = n_complete_years * samples_per_year
        yearly_data[year] = values_array[start_idx:]
    
    return yearly_data

def plot_comparison(model_data):
    yearly_data = {model: split_by_years(values) for model, values in model_data.items()}
    
    all_years = sorted(set().union(*[data.keys() for data in yearly_data.values()]))
    
    plt.figure(figsize=(14, 10))
    
    model_styles = {
        "Gemma-27b-ideal": {"color": "#2E86AB", "marker": "o", "label": "Gemma3-27b-ideal"},
        "Gemma-27b-generic": {"color": "#A23B72", "marker": "s", "label": "Gemma3-27b-generic"}, 
        "Gemma-27b-no-retrieval": {"color": "#F18F01", "marker": "^", "label": "Gemma3-27b-non-retrieval"}
    }
    
    for model_name in model_styles.keys():
        if model_name not in model_data:
            continue
            
        all_years_model = []
        all_values_model = []
        
        for year in all_years:
            if year in yearly_data[model_name]:
                values = yearly_data[model_name][year]
                year_jittered = np.random.normal(year, 0.1, len(values))
                all_years_model.extend(year_jittered)
                all_values_model.extend(values)
        
        plt.scatter(all_years_model, all_values_model, 
                   c=model_styles[model_name]["color"], 
                   marker=model_styles[model_name]["marker"],
                   alpha=0.6, s=100, 
                   label=model_styles[model_name]["label"])
        
        year_means = []
        years_with_data = []
        
        for year in all_years:
            if year in yearly_data[model_name]:
                values = yearly_data[model_name][year]
                year_means.append(np.mean(values))
                years_with_data.append(year)
        
        if len(years_with_data) > 1:
            plt.plot(years_with_data, year_means, 
                    color=model_styles[model_name]["color"], 
                    linewidth=3, alpha=0.8)
    
    plt.xlabel("Year", fontsize=32)
    plt.ylabel(r"LEA $A^{rag}$", fontsize=32)
    plt.xlim(min(all_years) - 0.5, max(all_years) + 0.5)
    plt.xticks(all_years, rotation=45, fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper right', fontsize=26)
    plt.tight_layout()
    plt.show()

def print_statistics(model_data):
    for model_name, values in model_data.items():
        values = np.array(values)
        yearly_data = split_by_years(values)
        
        print(f"\n{model_name}:")
        print(f"  Total samples: {len(values)}")
        print(f"  Mean: {np.mean(values):.2f}%")
        print(f"  Std: {np.std(values):.2f}%")
        print(f"  Range: {np.min(values):.2f}% - {np.max(values):.2f}%")
        
        for year in sorted(yearly_data.keys()):
            year_values = yearly_data[year]
            zero_count = np.sum(year_values == 0.0)
            print(f"    {year}: n={len(year_values)}, μ={np.mean(year_values):.1f}%, "
                  f"zeros={zero_count} ({zero_count/len(year_values)*100:.1f}%)")

def main():
    model_files = {
        "Gemma-27b-ideal": "./results/LEA/independence_results_filtered_x_ideal_theta_ideal_y_gemma.json",
        "Gemma-27b-generic": "./results/LEA/independence_results_filtered_x_generic_theta_generic_y_gemma.json",
        "Gemma-27b-no-retrieval": "./results/LEA/independence_results_filtered_x_no_theta_y_prime_gemma.json",
    }
    
    model_data = {}
    for model_name, file_path in model_files.items():
        try:
            model_data[model_name] = load_model_data(file_path)
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
    
    if len(model_data) >= 2:
        plot_comparison(model_data)
        print_statistics(model_data)
    else:
        print("Need at least 2 models for comparison!")

if __name__ == "__main__":
    main()