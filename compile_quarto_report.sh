#!/bin/bash

# Script to compile Quarto document to PDF
# This requires Quarto to be installed: https://quarto.org/docs/get-started/

# Print status
echo "Compiling executive summary report using Quarto..."

# Set the current directory to the project root
cd "$(dirname "$0")"

# Compile the report to PDF
quarto render results/executive_summary_report.qmd --to pdf

echo "Report compiled successfully to results/executive_summary_report.pdf"

# You can uncomment to also create an HTML version
# echo "Also creating HTML version..."
# quarto render results/executive_summary_report.qmd --to html
# echo "HTML version created at results/executive_summary_report.html"