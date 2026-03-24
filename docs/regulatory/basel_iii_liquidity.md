# Basel III Liquidity and Leverage Requirements

## Introduction

Basel_III introduced two quantitative liquidity standards and a non-risk-based Leverage_Ratio to complement the risk-based capital framework. Prior to Basel_III, the Basel_II framework did not include binding global liquidity requirements, a gap that the 2007-2009 financial crisis exposed with devastating consequences. The BCBS designed the LCR (Liquidity Coverage Ratio), NSFR (Net Stable Funding Ratio), and Leverage_Ratio to address short-term liquidity stress, longer-term structural funding mismatches, and excessive balance sheet leverage, respectively.

## LCR (Liquidity Coverage Ratio)

### Definition and Purpose

The LCR is a short-term liquidity metric introduced by Basel_III. The LCR requires banks to hold a sufficient stock of high-quality liquid assets (HQLA) to cover total net cash outflows over a 30-day stress scenario. The LCR is designed to ensure that banks can survive a severe short-term liquidity disruption, such as a sudden loss of wholesale funding, significant deposit withdrawals, or a rating downgrade.

The LCR formula is:

**LCR = Stock of HQLA / Total Net Cash Outflows over 30 days**

Basel_III requires a minimum LCR of 100%, meaning that a bank's HQLA must at least equal its projected net cash outflows over the 30-day stress period.

### HQLA Components

The BCBS defines HQLA in three tiers:

- **Level 1 assets**: The highest quality and most liquid assets, with no haircut applied. These include cash, central bank reserves, and certain sovereign debt securities with a 0% risk weight under the SA for Credit_Risk. Level 1 assets have no cap within the HQLA stock.

- **Level 2A assets**: High-quality assets subject to a 15% haircut. These include certain sovereign, central bank, and public sector entity debt securities rated AA- or higher, and certain corporate debt securities and covered bonds rated AA- or higher. Level 2A assets are capped at 40% of total HQLA.

- **Level 2B assets**: Moderate-quality assets subject to larger haircuts (25%-50%). These include certain corporate debt securities rated between A+ and BBB-, certain residential mortgage-backed securities, and certain equity shares. Level 2B assets are capped at 15% of total HQLA.

### Net Cash Outflows

Total net cash outflows are calculated as total expected cash outflows minus total expected cash inflows (capped at 75% of outflows) over the 30-day stress period. The BCBS specifies run-off factors for different categories of liabilities and draw-down factors for off-balance-sheet commitments. For example:

- Stable retail deposits: 5% run-off rate
- Less stable retail deposits: 10% run-off rate
- Unsecured wholesale funding from financial institutions: 100% run-off rate
- Committed credit and liquidity facilities to financial institutions: 40% run-off rate

### Implementation Timeline

The LCR was phased in starting January 2015 with a minimum requirement of 60%, increasing by 10 percentage points annually until reaching the full 100% requirement in January 2019. National supervisors, including the BCB, implemented the LCR schedule in line with their domestic frameworks.

## NSFR (Net Stable Funding Ratio)

### Definition and Purpose

The NSFR is a longer-term structural liquidity metric under Basel_III. While the LCR addresses short-term liquidity resilience, the NSFR promotes resilience over a one-year time horizon. The NSFR requires banks to maintain a stable funding profile relative to the composition of their assets and off-balance-sheet activities.

The NSFR formula is:

**NSFR = Available Stable Funding (ASF) / Required Stable Funding (RSF)**

Basel_III requires a minimum NSFR of 100%, meaning that a bank's available stable funding must at least cover its required stable funding.

### Available Stable Funding (ASF)

ASF is the portion of capital and liabilities expected to be reliable over a one-year horizon. The BCBS assigns ASF factors to different funding sources:

- Tier_1 and Tier_2 capital instruments: 100% ASF factor
- Stable retail deposits: 95% ASF factor
- Less stable retail deposits: 90% ASF factor
- Wholesale funding with residual maturity of one year or more: 100% ASF factor
- Wholesale funding with residual maturity less than six months from financial institutions: 0% ASF factor

### Required Stable Funding (RSF)

RSF represents the amount of stable funding required by a bank's assets and off-balance-sheet exposures. The BCBS assigns RSF factors based on the liquidity characteristics of assets:

- Cash and central bank reserves: 0% RSF factor
- Unencumbered Level 1 HQLA: 0% RSF factor (since they can be readily monetized)
- Performing loans to financial institutions with residual maturity less than six months: 10% RSF factor
- Performing loans to non-financial corporates with residual maturity of one year or more: 65% RSF factor
- Residential mortgages with residual maturity of one year or more and risk weight of 35% or less under SA: 65% RSF factor
- Other assets not included above: 100% RSF factor

### NSFR Interaction with Other Requirements

The NSFR complements the LCR by ensuring that banks do not rely excessively on short-term wholesale funding to finance longer-term assets. Together, LCR and NSFR form the two quantitative pillars of the Basel_III liquidity framework. The NSFR also interacts with the capital framework: Tier_1 and Tier_2 capital instruments receive a 100% ASF factor because they are by definition long-term or permanent sources of funding.

## Leverage_Ratio

### Definition and Purpose

The Leverage_Ratio is a non-risk-based capital measure introduced by Basel_III to constrain the build-up of excessive leverage in the banking system. The Leverage_Ratio serves as a backstop to the risk-based capital requirements (CAR, CET1 ratio), addressing the risk that internal models or standardised approaches may underestimate the true riskiness of certain exposures.

The Leverage_Ratio formula is:

**Leverage_Ratio = Tier_1 Capital / Total Exposure Measure**

Basel_III requires a minimum Leverage_Ratio of 3%. This means that a bank must hold Tier_1 capital equal to at least 3% of its total exposure measure.

### Total Exposure Measure

The total exposure measure includes:

- On-balance-sheet exposures (excluding items deducted from Tier_1 capital)
- Derivative exposures, calculated using the SA_CCR (Standardised Approach for Counterparty Credit Risk) methodology
- Securities financing transaction exposures
- Off-balance-sheet items, converted to credit equivalent amounts using specified credit conversion factors

Unlike RWA, the total exposure measure does not apply risk weights, ensuring that all exposures are captured regardless of perceived riskiness. This is the key distinction that makes the Leverage_Ratio a non-risk-based measure.

### G_SIB Leverage Buffer

Under Basel_III, G_SIB institutions are subject to an additional Leverage_Ratio buffer equal to 50% of their G_SIB_Surcharge. For example, a G_SIB with a G_SIB_Surcharge of 2% would face a Leverage_Ratio buffer of 1%, bringing its total Leverage_Ratio requirement to 4%. This buffer must be met with CET1 capital.

## Liquidity Monitoring Tools

In addition to the LCR and NSFR, the BCBS specified several monitoring tools under Basel_III to provide supervisors with additional information:

- **Contractual maturity mismatch**: Analysis of contractual cash flows across time bands to identify maturity transformation risks.
- **Concentration of funding**: Monitoring of significant funding sources and counterparty concentrations that could create vulnerability if withdrawn.
- **Available unencumbered assets**: Identification of assets that could serve as collateral for additional secured funding.
- **LCR by significant currency**: Banks must monitor LCR in each significant currency to ensure they can meet obligations in specific currencies.
- **Market-related monitoring tools**: Market data such as bank credit default swap spreads, equity prices, and wholesale funding rates that may signal emerging liquidity stress.

## Pillar_2 and Pillar_3 Interactions

Under Pillar_2, banks must conduct liquidity Stress_Testing beyond the standardized LCR scenario. The ICAAP (which extends to liquidity through the Internal Liquidity Adequacy Assessment Process, or ILAAP) requires banks to assess their liquidity positions under institution-specific and market-wide stress scenarios. The SREP conducted by supervisors evaluates these assessments and may impose additional liquidity requirements.

Under Pillar_3, banks must publicly disclose their LCR, NSFR, and Leverage_Ratio, along with qualitative information about their liquidity risk management frameworks. These disclosures, aligned with the BCBS_239 principles for risk data aggregation and reporting, enable market participants to assess the liquidity and leverage profiles of individual banks.

## Brazilian Implementation

The BCB has implemented the Basel_III liquidity and leverage requirements in Brazil, adapting the BCBS standards to the domestic banking system. Brazilian banks are required to maintain LCR at or above 100%, NSFR at or above 100%, and Leverage_Ratio at or above 3%. The BCB also sets the CCyB rate for Brazilian exposures, which interacts with the overall capital and liquidity framework.

## Conclusion

The LCR, NSFR, and Leverage_Ratio are fundamental pillars of the Basel_III framework established by the BCBS. The LCR ensures short-term liquidity resilience over a 30-day stress period, the NSFR promotes stable funding over a one-year horizon, and the Leverage_Ratio constrains excessive balance sheet leverage. Together with the risk-based capital requirements (CET1, Tier_1, Tier_2, and associated buffers), these metrics form a comprehensive prudential framework that addresses the key vulnerabilities exposed by the global financial crisis.
