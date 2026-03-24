# Basel III Capital Structure and Requirements

## Introduction

The capital framework under Basel_III, developed by the BCBS, establishes a rigorous hierarchy of capital instruments and minimum requirements designed to ensure that banks maintain sufficient loss-absorbing capacity. Basel_III significantly strengthened the quality, consistency, and transparency of the capital base compared to Basel_II. This document details the capital structure, minimum ratios, capital buffers, RWA calculation principles, and the CAR formula.

## Capital Components

### CET1 (Common Equity Tier 1)

CET1 is the highest quality component of regulatory capital under Basel_III. CET1 consists of common shares issued by the bank, retained earnings, accumulated other comprehensive income, and other disclosed reserves. CET1 is a component of Tier_1 capital and represents the core loss-absorbing capacity of a bank. Basel_III requires a minimum CET1 ratio of 4.5% of RWA. CET1 instruments must be perpetual, have no mandatory distributions, and absorb losses on a going-concern basis. The emphasis on CET1 quality was a direct response to the financial crisis, where many instruments previously counted as capital proved inadequate to absorb losses.

### AT1 (Additional Tier 1)

AT1 is the second component of Tier_1 capital. AT1 instruments include perpetual subordinated debt and certain preference shares that meet strict eligibility criteria defined by the BCBS under Basel_III. AT1 instruments must include a mechanism for loss absorption at the point of non-viability, either through conversion to common equity or write-down. AT1 is a component of Tier_1 capital alongside CET1. Basel_III requires that AT1 plus CET1 together achieve a minimum Tier_1 ratio of 6% of RWA. This means AT1 can contribute up to 1.5% of the 6% Tier_1 minimum.

### Tier_1 Capital

Tier_1 capital is the sum of CET1 and AT1. Tier_1 capital represents the primary measure of a bank's financial strength from a regulatory perspective. Basel_III requires a minimum Tier_1 ratio of 6% of RWA. Tier_1 capital must be available to absorb losses on a going-concern basis, meaning it can absorb losses while the bank continues operating. The Leverage_Ratio requirement (minimum 3%) is also calculated using Tier_1 capital as the numerator.

### Tier_2 Capital

Tier_2 is supplementary capital that provides loss absorption on a gone-concern basis (i.e., in the event of liquidation). Tier_2 instruments include subordinated debt with an original maturity of at least five years and certain loan loss provisions. Under Basel_III, the minimum total capital requirement (Tier_1 plus Tier_2) is 8% of RWA. This means Tier_2 can contribute up to 2% of the 8% total minimum. Tier_2 instruments must also include a loss-absorption mechanism at the point of non-viability.

## Capital Buffers

Basel_III introduced several capital buffers above the minimum requirements, all of which must be met with CET1 capital.

### CCB (Capital Conservation Buffer)

The CCB is a mandatory buffer of 2.5% of RWA that sits above the minimum capital requirements. The CCB must be composed entirely of CET1. When a bank's CET1 ratio falls into the CCB range (between 4.5% and 7.0%), automatic restrictions on distributions (dividends, share buybacks, and discretionary bonus payments) are triggered. The CCB was designed by the BCBS to ensure that banks build up capital during normal times that can be drawn down during periods of stress. The CCB is a fixed buffer that applies to all banks uniformly.

### CCyB (Countercyclical Capital Buffer)

The CCyB is a macroprudential tool that ranges from 0% to 2.5% of RWA (and potentially higher in some jurisdictions). The CCyB must be met with CET1 capital. National authorities, such as the BCB in Brazil, set the CCyB rate based on their assessment of systemic risk buildup in the credit cycle. When credit growth is excessive, the CCyB is activated to build additional resilience. During downturns, the CCyB can be released to support lending capacity. The CCyB is calculated as a weighted average of the buffer rates in jurisdictions where a bank has credit exposures.

### G_SIB_Surcharge

The G_SIB_Surcharge is an additional CET1 requirement imposed on G_SIB (Global Systemically Important Banks). The BCBS identifies G_SIB institutions annually based on criteria including size, interconnectedness, substitutability, complexity, and cross-jurisdictional activity. The G_SIB_Surcharge ranges from 1.0% to 3.5% of RWA (with an empty top bucket of 3.5% to discourage further systemic importance). The G_SIB_Surcharge must be met entirely with CET1. The rationale for the G_SIB_Surcharge is that the failure of a G_SIB institution would impose externalities on the global financial system beyond those of smaller banks.

### D_SIB_Buffer

The D_SIB_Buffer is an additional capital requirement for Domestically Systemically Important Banks. National supervisors, including the BCB, identify domestic systemically important institutions and set the D_SIB_Buffer level. The D_SIB_Buffer must be met with CET1 capital. The D_SIB_Buffer operates alongside the G_SIB_Surcharge; for banks designated as both G_SIB and D-SIB, the higher of the two surcharges typically applies.

## Total Capital Requirements Summary

For a non-systemically important bank, the total CET1 requirement under Basel_III is:

| Component | Minimum Ratio (% of RWA) |
|---|---|
| CET1 minimum | 4.5% |
| CCB | 2.5% |
| CCyB | 0% - 2.5% |
| **Total CET1 requirement** | **7.0% - 9.5%** |

For a G_SIB institution, the G_SIB_Surcharge adds an additional 1.0% to 3.5% of CET1, bringing the potential CET1 requirement to 10.5% to 13.0% of RWA.

The total capital stack is:

| Capital Layer | Minimum Ratio |
|---|---|
| CET1 | 4.5% |
| AT1 | 1.5% |
| Tier_1 (CET1 + AT1) | 6.0% |
| Tier_2 | 2.0% |
| Total Capital (Tier_1 + Tier_2) | 8.0% |
| CCB | 2.5% |
| CCyB | 0% - 2.5% |
| G_SIB_Surcharge | 1.0% - 3.5% |
| D_SIB_Buffer | Jurisdiction-specific |

## RWA (Risk-Weighted Assets) Calculation Overview

RWA is the denominator in the CAR formula and represents the total assets of a bank weighted according to their risk. Under Basel_III, RWA is calculated for three main risk categories:

- **Credit_Risk RWA**: Calculated using either the SA (Standardised Approach), where supervisory risk weights are applied, or the IRB approach (IRB_Foundation or IRB_Advanced), where banks use internal estimates of PD, LGD, and EAD. Basel_III final reforms introduced output floors requiring that RWA calculated under IRB cannot fall below 72.5% of RWA calculated under the SA. This addresses the concern that internal models may underestimate risk.

- **Market_Risk RWA**: Calculated under the FRTB framework, which replaced the previous VaR-based approach with a more comprehensive methodology. Market_Risk RWA captures risks from trading book positions.

- **Operational_Risk RWA**: Under Basel_III final reforms, Operational_Risk RWA is calculated using the SMA (Standardised Measurement Approach), which replaced the BIA and other previous methods. The SMA uses a combination of a bank's financial statements (business indicator) and historical operational loss data.

Total RWA equals Credit_Risk RWA plus Market_Risk RWA plus Operational_Risk RWA.

## CAR (Capital Adequacy Ratio) Formula

The CAR is the primary metric for assessing a bank's capital adequacy under Basel_III. The CAR formula is:

**CAR = (Tier_1 + Tier_2) / RWA**

Where:
- Tier_1 = CET1 + AT1
- Tier_2 = Eligible Tier_2 instruments
- RWA = Credit_Risk RWA + Market_Risk RWA + Operational_Risk RWA

Basel_III requires a minimum CAR of 8%, before the application of buffers. When buffers are included, the effective minimum capital requirement rises substantially.

Similarly, the CET1 ratio is:

**CET1 Ratio = CET1 / RWA**

And the Tier_1 ratio is:

**Tier_1 Ratio = Tier_1 / RWA**

## Supervisory Dimensions

Under Pillar_2, the ICAAP process requires banks to assess whether their capital levels are adequate for their specific risk profiles, potentially resulting in capital requirements above the Pillar_1 minimums. The SREP conducted by supervisors evaluates the quality of ICAAP and may impose additional Pillar_2 capital add-ons. Stress_Testing under Pillar_2 evaluates whether a bank's capital would remain above minimum requirements under adverse macroeconomic scenarios.

Under Pillar_3, banks must disclose their capital composition, CAR, CET1 ratio, Tier_1 ratio, and the details of each buffer requirement. These disclosures enable market participants to assess the capital strength of individual banks, supporting market discipline.

## Conclusion

The Basel_III capital framework, designed by the BCBS, creates a multi-layered system of capital requirements that ensures banks hold sufficient high-quality capital to absorb losses. The hierarchy from CET1 through AT1, Tier_1, and Tier_2, combined with buffers such as the CCB, CCyB, G_SIB_Surcharge, and D_SIB_Buffer, provides both microprudential and macroprudential protection. The CAR, calculated as total capital divided by RWA, serves as the central metric for regulatory capital adequacy assessment.
