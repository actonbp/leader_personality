#!/usr/bin/env Rscript

# Script to compile the executive summary report to PDF
# This requires R with knitr, rmarkdown, and kableExtra packages

# Install required packages if not already installed
if (!require("knitr")) install.packages("knitr")
if (!require("rmarkdown")) install.packages("rmarkdown")
if (!require("kableExtra")) install.packages("kableExtra")

# Load required libraries
library(knitr)
library(rmarkdown)
library(kableExtra)

# Set working directory to the project root
# Comment this out if running from the project root already
# setwd("/Users/bryanacton/Documents/GitHub/leader_personality")

# Compile the report
cat("Compiling executive summary report to PDF...\n")
render(
  "results/executive_summary_report.md",
  output_format = "pdf_document",
  output_file = "CEO_Personality_Analysis_Report.pdf",
  output_dir = "results"
)

cat("Report compiled successfully to results/CEO_Personality_Analysis_Report.pdf\n")