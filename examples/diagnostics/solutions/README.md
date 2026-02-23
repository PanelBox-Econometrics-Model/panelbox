# Diagnostics Tutorial Solutions

**Version:** 1.0.0
**Last Updated:** 2026-02-22

## Purpose

This directory contains complete solutions to all exercises in the diagnostics tutorial notebooks. Solutions are provided to:

- Help you verify your work after attempting exercises independently
- Provide alternative approaches and best practices
- Offer detailed explanations and interpretation guidance
- Demonstrate common pitfalls and how to avoid them

---

## Important: Try Exercises First

**Do not look at solutions before attempting the exercises yourself.** Working through the problems independently is essential for building understanding. The struggle of debugging and reasoning through problems is where the deepest learning occurs.

### When to Consult Solutions

**Good reasons:**
- You have genuinely tried and are stuck after 20+ minutes
- Your code works but you want to verify your interpretation
- You want to see alternative approaches after solving the problem
- You are reviewing material you have already learned

**Not-so-good reasons:**
- You have not attempted the exercise yet
- You want to save time (shortcuts reduce learning)
- The exercise feels frustrating (that feeling is learning happening)

---

## Solution Files

| Solution | Tutorial | Topics Covered |
|----------|----------|----------------|
| `01_unit_root_tests_solutions.ipynb` | Unit Root Tests | LLC, IPS, Breitung, Hadri, CIPS implementations and interpretation |
| `02_cointegration_tests_solutions.ipynb` | Cointegration Tests | Pedroni, Kao, Westerlund tests; spurious regression detection |
| `03_specification_tests_solutions.ipynb` | Specification Tests | Hausman, Mundlak, Breusch-Pagan, serial correlation, CD tests |
| `04_spatial_diagnostics_solutions.ipynb` | Spatial Diagnostics | Moran's I, LM tests, weight matrix validation, panel spatial tests |

---

## Solution Format

Each solution notebook follows a consistent structure:

### 1. Exercise Restatement
Clear statement of what was asked, including the specific question or task.

### 2. Solution Approach
Explanation of the strategy and reasoning before showing any code.

### 3. Complete Code
Fully functional, well-commented solution code that can be run independently.

### 4. Output and Interpretation
Expected results with detailed statistical interpretation, including:
- What the test statistic and p-value mean
- How to state the conclusion in plain language
- What the result implies for model specification

### 5. Alternative Approaches
Other valid ways to solve the problem, when applicable.

### 6. Common Mistakes
Pitfalls to watch out for and how to recognize them (e.g., applying first-generation unit root tests when cross-sectional dependence is present).

### 7. Extensions
How to take the exercise further for deeper understanding.

---

## How to Use Solutions

### Recommended Approach

1. **Read the exercise** in the main tutorial notebook carefully
2. **Think about the approach** before writing any code
3. **Write your code** and run it
4. **Debug and iterate** until you get reasonable results
5. **Only then** open the corresponding solution notebook
6. **Compare approaches**: Your solution may be equally valid
7. **Understand differences**: Why did the solution take a different approach?

### Comparing Your Results

Your solution is correct if:
- Code runs without errors
- Test statistics and p-values are qualitatively similar
- Conclusions (reject/fail to reject) match
- Interpretation is economically and statistically sound

Do not worry if:
- Your code structure differs from the solution
- You used different variable names or formatting
- Your plots have different styling
- You took more or fewer steps to reach the answer

### Red Flags (Check Against Solutions)

- Test conclusions are opposite (reject vs. fail to reject)
- Test statistics differ by orders of magnitude
- You cannot explain what a result means
- Your code produces warnings about convergence or singularity

---

## Tips for Learning from Solutions

### Active Reading Strategy

When reading solutions:
1. **Predict**: What do you expect before running the code?
2. **Run**: Execute the solution code
3. **Compare**: How do results match your expectations?
4. **Explain**: Can you explain why this approach works?
5. **Modify**: What happens if you change parameters or data?

### Consolidation

After reviewing solutions:
1. **Reproduce** the solution without looking at it
2. **Extend** the analysis with new questions
3. **Apply** the technique to a different dataset
4. **Explain** the solution to someone else (or yourself aloud)

---

For questions about solutions, consult the main tutorial notebook first, then open an issue on the PanelBox GitHub repository.
