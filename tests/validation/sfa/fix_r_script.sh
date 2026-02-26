#!/bin/bash
# Fix gammaParm and sigmaSq references in generate_r_frontier_results.R

# Create backup
cp generate_r_frontier_results.R generate_r_frontier_results.R.bak

# Fix panel Pitt-Lee section (lines ~165-175)
sed -i '165,177s/sfa_pl\$sigmaSq/coef(sfa_pl)["sigmaSq"]/g' generate_r_frontier_results.R
sed -i '165,177s/sfa_pl\$gammaParm/coef(sfa_pl)["gamma"]/g' generate_r_frontier_results.R

# Fix panel BC92 section (lines ~233-245)
sed -i '233,245s/sfa_bc92\$sigmaSq/coef(sfa_bc92)["sigmaSq"]/g' generate_r_frontier_results.R
sed -i '233,245s/sfa_bc92\$gammaParm/coef(sfa_bc92)["gamma"]/g' generate_r_frontier_results.R

echo "Fixed R script"
