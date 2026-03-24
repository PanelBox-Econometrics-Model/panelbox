# Basel III Credit Risk Framework

## Introduction

Credit_Risk is the most significant source of risk for most banks and, accordingly, accounts for the largest share of RWA (Risk-Weighted Assets) under Basel_III. The BCBS has developed a comprehensive framework for Credit_Risk capital requirements that offers banks two primary approaches: the SA (Standardised Approach) and the IRB (Internal Ratings-Based) approach. The IRB approach is further divided into IRB_Foundation and IRB_Advanced. This document details these approaches, the key risk parameters (PD, LGD, EAD), the Expected_Loss formula, Credit_Risk capital calculation principles, and SA_CCR for counterparty credit risk.

## Credit Risk Approaches Under Basel III

### SA (Standardised Approach) for Credit Risk

The SA for Credit_Risk uses supervisory-determined risk weights based on external credit ratings or fixed regulatory categories. Under Basel_III, the BCBS revised the SA to increase its risk sensitivity compared to the Basel_II version while reducing mechanistic reliance on external ratings.

Key features of the revised SA under Basel_III include:

- **Exposures to banks**: Risk weights ranging from 20% to 150% depending on external ratings, or alternatively based on the Standardised Credit Risk Assessment Approach (SCRA) where external ratings are not available.
- **Exposures to corporates**: Risk weights from 20% to 150% based on external ratings. Unrated corporate exposures generally receive a 100% risk weight.
- **Retail exposures**: Regulatory retail exposures receive a 75% risk weight. Residential mortgage exposures receive risk weights based on loan-to-value ratios, ranging from 20% to 70%.
- **Real estate exposures**: Basel_III introduced more granular risk weights for real estate, distinguishing between income-producing real estate and other categories.
- **Sovereign exposures**: Risk weights from 0% to 150% based on external ratings, with national discretion for domestic sovereign exposures.

The RWA under SA is calculated as: **RWA = Exposure Amount x Risk Weight**

The SA serves as both a standalone approach for banks that do not use internal models and as the basis for the output floor applied to IRB banks under Basel_III final reforms.

### IRB (Internal Ratings-Based) Approach

The IRB approach allows qualifying banks, with supervisory approval, to use their own internal models to estimate key risk parameters for Credit_Risk capital calculation. The IRB approach was introduced under Basel_II and retained with modifications under Basel_III. There are two variants:

#### IRB_Foundation

Under IRB_Foundation, banks estimate PD (Probability of Default) using their own internal models, while LGD (Loss Given Default) and EAD (Exposure at Default) are set by the BCBS at supervisory prescribed values. IRB_Foundation is designed for banks that have developed credit rating systems but may not have sufficient data or modelling capability for all risk parameters.

Supervisory values under IRB_Foundation include:
- LGD for senior unsecured claims: 45%
- LGD for subordinated claims: 75%
- EAD for on-balance-sheet items: the drawn amount
- EAD for off-balance-sheet items: prescribed credit conversion factors

#### IRB_Advanced

Under IRB_Advanced, banks estimate all three key risk parameters (PD, LGD, and EAD) using their own internal models. IRB_Advanced provides the greatest risk sensitivity but requires the most sophisticated data infrastructure, modelling capability, and governance. Banks using IRB_Advanced must demonstrate to supervisors that their models are robust, well-validated, and based on sufficient historical data.

Basel_III final reforms introduced restrictions on IRB_Advanced:
- For exposures to large corporates (consolidated revenues exceeding EUR 500 million) and exposures to financial institutions, banks must use IRB_Foundation rather than IRB_Advanced.
- For equity exposures, neither IRB_Foundation nor IRB_Advanced is permitted; the SA must be used.
- Input floors were introduced for PD (minimum 0.05% for most asset classes), LGD (minimum 25% for unsecured corporate exposures), and EAD parameters.

### Output Floor

A critical Basel_III final reform is the output floor, which requires that total RWA calculated under the IRB approach cannot be less than 72.5% of total RWA calculated under the SA. This floor was introduced to address concerns that internal models could produce excessively low capital requirements. The output floor is being phased in from 2023 to 2028.

## Key Risk Parameters

### PD (Probability of Default)

PD is the likelihood that a borrower will default on its obligations within a one-year horizon. Under the IRB approach, PD is the fundamental input to the Credit_Risk capital function. Banks estimate PD using statistical models calibrated on historical default data. PD estimates must represent long-run averages through an economic cycle, not point-in-time estimates. Under Basel_III, a minimum PD floor of 0.05% applies for most exposure classes to prevent unrealistically low estimates.

### LGD (Loss Given Default)

LGD is the proportion of the exposure that is lost if a default occurs, taking into account recoveries from collateral, guarantees, and the workout process. LGD is expressed as a percentage of EAD. Under IRB_Foundation, LGD values are prescribed by the BCBS. Under IRB_Advanced, banks estimate LGD using their own models based on historical loss data. LGD estimates must reflect downturn economic conditions (downturn LGD) to ensure conservatism.

### EAD (Exposure at Default)

EAD is the total exposure amount at the time of default. For on-balance-sheet items, EAD is typically the outstanding balance. For off-balance-sheet items, EAD includes an estimate of additional drawdowns that may occur before default, using credit conversion factors. Under IRB_Advanced, banks estimate EAD internally, while under IRB_Foundation, prescribed conversion factors are used.

## Expected_Loss Formula

Expected_Loss is the average loss a bank expects to incur on its credit portfolio over a one-year period. The Expected_Loss formula is:

**Expected_Loss = PD x LGD x EAD**

Expected_Loss represents the mean of the loss distribution and is a key input for provisioning and credit pricing decisions. Under Basel_III, Expected_Loss is not directly a capital charge; instead, it is compared against the bank's accounting provisions. If Expected_Loss exceeds provisions, the shortfall is deducted from CET1 capital. If provisions exceed Expected_Loss, the excess can be added to Tier_2 capital (up to a cap).

The Credit_Risk capital requirement under IRB is designed to cover unexpected losses beyond Expected_Loss, calibrated to a 99.9% confidence level over a one-year horizon. This means the capital requirement represents the loss that would be exceeded only once in 1,000 years on average.

## Credit Risk Capital Calculation Under IRB

The IRB capital formula for Credit_Risk uses a single-factor Gaussian copula model (the Vasicek model). The key inputs are PD, LGD, EAD, and the asset correlation parameter (which is a supervisory function of PD). The formula produces a capital requirement per exposure that, when multiplied by EAD, gives the capital charge.

The capital requirement (K) for a given exposure under IRB is:

**K = [LGD x N[(1-R)^(-0.5) x G(PD) + (R/(1-R))^0.5 x G(0.999)] - PD x LGD] x Maturity Adjustment**

Where:
- N[] is the cumulative standard normal distribution
- G() is the inverse cumulative standard normal distribution
- R is the asset correlation parameter
- The maturity adjustment accounts for the exposure's remaining term

RWA for the exposure equals: **RWA = K x 12.5 x EAD**

The factor 12.5 is the reciprocal of the 8% minimum CAR, ensuring consistency with the capital ratio framework.

## SA_CCR (Standardised Approach for Counterparty Credit Risk)

SA_CCR is the method prescribed by Basel_III for calculating the EAD of derivative exposures for counterparty Credit_Risk purposes. SA_CCR replaced the previous Current Exposure Method (CEM) and was designed to be more risk-sensitive while remaining simpler than internal model approaches.

Under SA_CCR, the EAD for a netting set of derivatives is:

**EAD = alpha x (Replacement Cost + Potential Future Exposure)**

Where alpha is a supervisory parameter set at 1.4. The replacement cost represents the current mark-to-market loss if the counterparty defaults today. The potential future exposure (PFE) represents the possible increase in exposure over the margin period of risk.

SA_CCR is also used in the Leverage_Ratio calculation for derivative exposures and in the determination of the CVA (Credit Valuation Adjustment) capital charge for counterparty Credit_Risk.

## Relationship with Other Framework Components

Credit_Risk capital requirements under Pillar_1 interact with Pillar_2 through the ICAAP, where banks must assess whether their Credit_Risk capital is adequate for their specific portfolios, including concentration risk and other factors not fully captured by Pillar_1 models. The SREP evaluates these assessments. Stress_Testing under Pillar_2 examines how Credit_Risk parameters (PD, LGD, EAD) would deteriorate under adverse scenarios and the consequent impact on capital adequacy.

Under Pillar_3, banks must disclose their Credit_Risk exposures by approach (SA vs. IRB), asset class, geography, and industry. IRB banks must disclose their PD, LGD, and EAD estimates, along with model validation results.

## Conclusion

The Basel_III Credit_Risk framework, developed by the BCBS, provides a sophisticated system for calculating capital requirements. Whether using the SA or the IRB approach (IRB_Foundation or IRB_Advanced), banks must ensure they hold sufficient capital to cover unexpected credit losses. The key risk parameters PD, LGD, and EAD form the foundation of credit risk measurement, with Expected_Loss = PD x LGD x EAD representing the expected annual loss. SA_CCR addresses the specialized challenge of counterparty Credit_Risk in derivative portfolios. Together, these components ensure that Credit_Risk, as the dominant source of banking risk, is adequately capitalized under Basel_III.
