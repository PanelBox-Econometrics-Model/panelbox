// MathJax configuration for PanelBox documentation

window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
    tags: 'ams',
    tagSide: 'right',
    tagIndent: '0.8em',
    multlineWidth: '85%',
    // Support for Greek letters and common symbols
    macros: {
      // Common econometrics notation
      E: "\\mathbb{E}",
      Var: "\\mathrm{Var}",
      Cov: "\\mathrm{Cov}",
      Corr: "\\mathrm{Corr}",
      plim: "\\operatorname{plim}",
      // Greek letters shortcuts
      eps: "\\varepsilon",
      // Common operators
      argmin: "\\operatorname{argmin}",
      argmax: "\\operatorname{argmax}"
    }
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  },
  loader: {
    load: ['[tex]/ams']
  },
  svg: {
    fontCache: 'global',
    displayAlign: 'left',
    displayIndent: '2em'
  }
};

document$.subscribe(() => {
  MathJax.typesetPromise()
})
