# Basel III Market Risk, Operational Risk, and IRRBB

## Introduction

Beyond Credit_Risk, Basel_III establishes comprehensive capital requirements for Market_Risk and Operational_Risk under Pillar_1. The BCBS undertook fundamental reforms in both areas following the financial crisis, culminating in the FRTB (Fundamental Review of the Trading Book) for Market_Risk and the SMA (Standardised Measurement Approach) for Operational_Risk. This document covers the Market_Risk framework including VaR and CVA, the FRTB overhaul, Operational_Risk approaches (BIA and SMA), and IRRBB (Interest Rate Risk in the Banking Book).

## Market_Risk Framework

### Definition and Scope

Market_Risk is the risk of losses in on- and off-balance-sheet positions arising from adverse movements in market prices. Market_Risk encompasses interest rate risk, equity risk, foreign exchange risk, commodity risk, and credit spread risk in the trading book. Under Basel_III, the boundary between the trading book and the banking book is more strictly defined than under Basel_II, reducing opportunities for regulatory arbitrage.

### VaR (Value at Risk)

VaR has been a central risk metric for Market_Risk measurement since the 1996 amendment to Basel I. VaR estimates the maximum potential loss on a portfolio over a specified time horizon at a given confidence level. Under the pre-FRTB framework:

- **Regulatory VaR**: Calculated at a 99% confidence level over a 10-day holding period.
- **Stressed VaR**: Introduced under Basel_III (Basel 2.5), calculated using the same parameters but calibrated to a 12-month period of significant financial stress.
- **Capital charge**: The Market_Risk capital charge was the higher of the previous day's VaR/Stressed VaR or the average over the preceding 60 business days, multiplied by a supervisory multiplier (minimum 3).

While VaR remains an important risk management tool, the BCBS identified several shortcomings: VaR does not capture tail risk beyond the confidence level, it is not subadditive (violating a desirable property of coherent risk measures), and it can be manipulated through model assumptions.

### FRTB (Fundamental Review of the Trading Book)

The FRTB represents the BCBS's comprehensive overhaul of Market_Risk capital requirements under Basel_III. FRTB was finalized in January 2019 and is a key component of the Basel_III final reforms. FRTB addresses the limitations of VaR and introduces a more robust framework.

Key features of the FRTB include:

**Revised boundary between trading and banking books**: FRTB establishes clearer criteria for which instruments belong in the trading book versus the banking book. Instruments must be held for trading intent, and switching between books is subject to strict capital penalties.

**Standardised Approach (SA) for Market Risk**: The FRTB SA uses a sensitivity-based approach where capital charges are calculated based on risk factor sensitivities (delta, vega, and curvature) across prescribed risk classes. This replaced the simpler risk-weight-based approach of the previous SA.

**Internal Models Approach (IMA)**: The FRTB IMA replaces VaR with Expected Shortfall (ES) as the primary risk metric. ES is calculated at a 97.5% confidence level, which provides equivalent coverage to 99% VaR but better captures tail risk. ES must be calibrated to a stressed period. The IMA operates at the desk level, with approval required for each trading desk.

**Default Risk Charge (DRC)**: A separate capital charge for jump-to-default risk in the trading book, applicable under both the SA and IMA. The DRC captures the risk of sudden defaults of issuers in the trading portfolio.

**Residual Risk Add-On (RRAO)**: An additional charge for exotic instruments whose risks are not fully captured by the sensitivity-based SA.

### CVA (Credit Valuation Adjustment)

CVA is the market value of counterparty Credit_Risk on derivative portfolios. Under Basel_III, banks must hold capital against CVA risk, reflecting the potential for mark-to-market losses due to deterioration in counterparty creditworthiness (as opposed to outright default, which is captured by counterparty Credit_Risk capital via SA_CCR).

The BCBS introduced a revised CVA framework as part of the Basel_III final reforms:

- **Basic Approach (BA-CVA)**: A standardised formula-based approach that calculates CVA capital based on notional amounts, supervisory risk weights, and hedging positions. Available to all banks.
- **Standardised Approach (SA-CVA)**: A more risk-sensitive approach based on the sensitivity methodology from the FRTB, applied to the CVA portfolio. Requires supervisory approval.

CVA capital requirements interact with the SA_CCR framework (used to calculate EAD for counterparty Credit_Risk) and with Market_Risk capital for hedges of CVA.

## Operational_Risk Framework

### Definition

Operational_Risk is defined by the BCBS as the risk of loss resulting from inadequate or failed internal processes, people, and systems, or from external events. This includes legal risk but excludes strategic and reputational risk. Under Basel_III, Operational_Risk capital requirements are a component of Pillar_1 alongside Credit_Risk and Market_Risk.

### BIA (Basic Indicator Approach)

The BIA was the simplest approach for calculating Operational_Risk capital under Basel_II. Under BIA, the capital charge equals 15% of the average annual gross income over the previous three years (excluding years where gross income was negative). While the BIA was straightforward to implement, it was criticized for its lack of risk sensitivity: it did not account for a bank's operational loss history or the quality of its risk management practices. Under Basel_III final reforms, BIA has been superseded by the SMA for internationally active banks, although some jurisdictions may retain simplified approaches for smaller banks.

### SMA (Standardised Measurement Approach)

The SMA is the single Operational_Risk capital approach under the Basel_III final reforms, replacing the BIA, the Standardised Approach, and the Advanced Measurement Approaches. The SMA was designed by the BCBS to be more risk-sensitive than BIA while addressing the issues of comparability and complexity that plagued the Advanced Measurement Approaches.

The SMA has two components:

1. **Business Indicator Component (BIC)**: A financial-statement-based proxy for operational risk derived from three components: interest, leases, and dividends component (ILDC); services component (SC); and financial component (FC). The Business Indicator is calculated from these components, and marginal coefficients (12%, 15%, 18%) are applied to increasing buckets of the Business Indicator to produce the BIC.

2. **Internal Loss Multiplier (ILM)**: For banks in the higher Business Indicator buckets, the ILM adjusts the BIC based on the bank's average historical operational losses over the preceding 10 years. Banks with higher historical losses relative to their BIC will face an ILM greater than 1, increasing their capital charge. Conversely, banks with lower historical losses may benefit from an ILM below 1 (subject to national discretion on whether the ILM can reduce capital below the BIC).

The SMA Operational_Risk capital charge = BIC x ILM.

## IRRBB (Interest Rate Risk in the Banking Book)

### Definition

IRRBB is the risk to a bank's capital and earnings arising from adverse movements in interest rates affecting its banking book positions. Unlike Market_Risk in the trading book, IRRBB arises from maturity and repricing mismatches between assets and liabilities, basis risk, and optionality in banking products.

### Regulatory Treatment

Under Basel_III, the BCBS maintains IRRBB as a Pillar_2 risk rather than a Pillar_1 capital charge, reflecting the heterogeneity of banking book positions across institutions and jurisdictions. However, the BCBS published updated standards for IRRBB in 2016, requiring:

- Banks must measure IRRBB under six prescribed interest rate shock scenarios (parallel up, parallel down, steepener, flattener, short rates up, short rates down).
- Banks must assess both the economic value of equity (EVE) and net interest income (NII) impacts.
- An outlier test: if the decline in EVE under any scenario exceeds 15% of Tier_1 capital, supervisors should consider whether the bank's capital is adequate. This may trigger additional Pillar_2 capital requirements through the SREP process.

IRRBB interacts with the ICAAP, where banks must incorporate IRRBB assessment into their comprehensive capital adequacy evaluation. Stress_Testing scenarios should include adverse interest rate movements and their impact on bank profitability and capital.

## Integration Across Risk Types

Under Pillar_1, total RWA is the sum of Credit_Risk RWA, Market_Risk RWA, and Operational_Risk RWA. The CAR is then calculated as:

**CAR = (Tier_1 + Tier_2) / (Credit_Risk RWA + Market_Risk RWA + Operational_Risk RWA)**

This must meet the minimum of 8% (before buffers). CET1 must meet 4.5% of total RWA across all risk types.

Under Pillar_2, the ICAAP requires banks to assess capital adequacy across all risk types, including those not fully captured under Pillar_1 (such as IRRBB, concentration risk, and strategic risk). The SREP evaluates these assessments holistically.

Under Pillar_3, banks must disclose Market_Risk and Operational_Risk exposures, capital requirements, and methodologies. BCBS_239 principles require that risk data for all risk types be aggregated accurately and reported in a timely manner.

## Conclusion

The Basel_III framework, as developed by the BCBS, provides comprehensive Pillar_1 capital requirements for Market_Risk (through the FRTB, including VaR replacement by Expected Shortfall, the CVA framework, and SA_CCR), Operational_Risk (through the SMA, replacing the BIA and other approaches), and the Pillar_2 treatment of IRRBB. These frameworks interact with Credit_Risk capital requirements and with the broader supervisory and disclosure pillars to form a cohesive regulatory architecture that addresses the full spectrum of banking risks.
