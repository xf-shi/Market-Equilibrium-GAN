import pandas as pd

def csv_to_latex_table(csv_file, output_file=None, table_name = "10agents_power2"):
    """
    Reads a CSV file and converts its contents to LaTeX table code.
    Numbers are converted to scientific notation in the format of 1.00 x 10^3.
    
    Args:
        csv_file (str): Path to the input CSV file.
        output_file (str, optional): Path to save the generated LaTeX code. If None, prints the LaTeX code.
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)
    df = df[[x for x in df.columns if x != "S0"]]
    df.columns = sorted([f"1 {x}" if "Utility" in x else f"2 {x}" if "Loss" in x else f"3 {x}" for x in list(df.columns)])
    df.columns = [x[2:] for x in df.columns]
    
    prefixes = []
    suffixes = []
    for col in df.columns:
        if "Mean" in col:
            prefix, suffix = col.strip("Mean").strip(), "Mean"
        elif "SE" in col:
            prefix, suffix = col.strip("SE").strip(), "S.E"
        else:
            prefix, suffix = col, " "
        prefixes.append(prefix)
        suffixes.append(suffix)
    
    # Start building the LaTeX table
    latex_code = []
    latex_code.append(r"\begin{table}[ht]")
    latex_code.append(r"    \centering")
    latex_code.append(r"    \begin{tabular}{|" + "|".join(["c"] * len(df.columns)) + r"|}")
    latex_code.append(r"        \hline")
    
    # Add the table header
#    header = " & ".join(df.columns) + r" \\"
#    latex_code.append(f"        {header}")
    header_top = " & ".join([x if x == "Type" else f"\multicolumn{{2}}{{c|}}{{{x}}}" for x in prefixes[::2]]) + r" \\"
    latex_code.append(f"        {header_top}")
    header_bottom = " & ".join(suffixes) + r" \\"
    latex_code.append(f"        {header_bottom}")
    latex_code.append(r"        \hline")
    
    # Add the table rows
    for _, row in df.iterrows():
        row_data = []
        for value in row:
            if isinstance(value, (int, float)):
                # Convert numbers to scientific notation in LaTeX format
                formatted_value = f"{value:.2e}"
                base, exponent = formatted_value.split("e")
                base = float(base)
                exponent = int(exponent)
                latex_value = f"${base:.2f} \\times 10^{{{exponent}}}$"
                row_data.append(latex_value)
            else:
                # Keep non-numerical values as is
                row_data.append(str(value))
        latex_code.append("        " + " & ".join(row_data) + r" \\")
        latex_code.append(r"        \hline")
    
    # Close the LaTeX table
    latex_code.append(r"    \end{tabular}")
    if "power2" in table_name:
        caption = "Quadratic Cost "
        label = "quad-"
    else:
        caption = "Power 1.5 Cost "
        label = "power-"
    n_agents = int(table_name.split("_")[0].strip("agents"))
    caption += f"{n_agents} Agents"
    label += str(n_agents)
    latex_code.append(f"    \caption{{{caption}}}")
    latex_code.append(f"    \label{{tab:{label}}}")
    latex_code.append(r"\end{table}")
    
    # Join the LaTeX code into a single string
    latex_code_str = "\n".join(latex_code)
    
    # Save or print the LaTeX code
    if output_file:
        with open(output_file, 'w') as file:
            file.write(latex_code_str)
            file.write("\n\n")
        print(f"LaTeX table code saved to {output_file}")
    else:
        print(latex_code_str)
    
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)
    df = df[["S0", "Type"]]
    
    # Start building the LaTeX table
    latex_code = []
    latex_code.append(r"\begin{table}[ht]")
    latex_code.append(r"    \centering")
    latex_code.append(r"    \begin{tabular}{|" + "|".join(["c"] * len(df.columns)) + r"|}")
    latex_code.append(r"        \hline")
    
    # Add the table header
#    header = " & ".join(df.columns) + r" \\"
#    latex_code.append(f"        {header}")
    header_top = " & ".join(list(df.columns)) + r" \\"
    latex_code.append(f"        {header_top}")
    latex_code.append(r"        \hline")
    
    # Add the table rows
    for _, row in df.iterrows():
        row_data = []
        for value in row:
            if isinstance(value, (int, float)):
                # Convert numbers to scientific notation in LaTeX format
                formatted_value = f"{value:.2e}"
                base, exponent = formatted_value.split("e")
                base = float(base)
                exponent = int(exponent)
                latex_value = f"${base:.2f} \\times 10^{{{exponent}}}$"
                row_data.append(latex_value)
            else:
                # Keep non-numerical values as is
                row_data.append(str(value))
        latex_code.append("        " + " & ".join(row_data) + r" \\")
        latex_code.append(r"        \hline")
    
    # Close the LaTeX table
    latex_code.append(r"    \end{tabular}")
    if "power2" in table_name:
        caption = "Quadratic Cost "
        label = "quad-s0-"
    else:
        caption = "Power 1.5 Cost "
        label = "power-s0-"
    n_agents = int(table_name.split("_")[0].strip("agents"))
    caption += f"{n_agents} Agents $S_0$"
    label += str(n_agents)
    latex_code.append(f"    \caption{{{caption}}}")
    latex_code.append(f"    \label{{tab:{label}}}")
    latex_code.append(r"\end{table}")
    
    # Join the LaTeX code into a single string
    latex_code_str = "\n".join(latex_code)
    
    # Save or print the LaTeX code
    if output_file:
        with open(output_file, 'a') as file:
            file.write(latex_code_str)
        print(f"LaTeX table code saved to {output_file}")
    else:
        print(latex_code_str)

# Example usage
for table_name in ["5agents_power2", "10agents_power2", "2agents_power1.5", "5agents_power1.5"]:
    csv_file = f"Tables/{table_name}.csv"  # Replace with your CSV file path
    output_file = f"Tables/{table_name}.txt"  # Replace with desired output file path, or set to None to print
    csv_to_latex_table(csv_file, output_file, table_name = table_name)

