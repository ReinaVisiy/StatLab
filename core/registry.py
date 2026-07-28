"""
Central registry mapping every statistical law/test to its backend function
and the UI metadata needed to render its Level 2 card, Level 3 data-entry
page, and results page. This is the single source of truth for navigation
so that Level 2/3 page rendering can be generic (per project rule: zero
duplicate logic).
"""

# ---------------------------------------------------------------------------
# Entry-pattern codes (see spec section 3, DATA ENTRY PATTERN):
#   D = probability law calculator (params + query_type + k/a/b)
#   A = summary-stats OR raw-data single/two-group parametric test
#   B = one dataset per group, multiple groups (table, ragged columns ok)
#   C = bivariate (X, Y) data
#   G = goodness-of-fit (single sample + distribution-specific params,
#       optionally class_edges for continuous laws)
#   M = matrix/contingency-style custom input (chi-sq independence,
#       correlation matrix, two-way ANOVA, multinomial GOF)
# ---------------------------------------------------------------------------

DISCRETE_LAW_PARAMS = {
    "bernoulli": [{"name": "p", "label": "p (probability of success)", "type": "prob", "default": 0.5}],
    "binomial": [
        {"name": "n", "label": "n (number of trials)", "type": "posint", "default": 10},
        {"name": "p", "label": "p (probability of success)", "type": "prob", "default": 0.5},
    ],
    "poisson": [{"name": "mu", "label": "λ (mean rate)", "type": "pos", "default": 3.0}],
    "geometric": [{"name": "p", "label": "p (probability of success)", "type": "prob", "default": 0.5}],
    "negative_binomial": [
        {"name": "r", "label": "r (number of successes)", "type": "posint", "default": 3},
        {"name": "p", "label": "p (probability of success)", "type": "prob", "default": 0.5},
    ],
    "hypergeometric": [
        {"name": "M", "label": "M (population size)", "type": "posint", "default": 50},
        {"name": "n", "label": "n (number of success states in population)", "type": "posint", "default": 20},
        {"name": "N", "label": "N (sample size drawn)", "type": "posint", "default": 10},
    ],
    "discrete_uniform": [
        {"name": "a", "label": "a (lower bound)", "type": "int", "default": 1},
        {"name": "b", "label": "b (upper bound)", "type": "int", "default": 6},
    ],
    "multinomial": [
        {"name": "n", "label": "n (number of trials)", "type": "posint", "default": 10},
        {"name": "p", "label": "p (category probabilities, comma-separated)", "type": "vec", "default": "0.2,0.3,0.5"},
        {"name": "x", "label": "x (observed counts per category, comma-separated)", "type": "vec", "default": "2,3,5"},
    ],
}

CONTINUOUS_LAW_PARAMS = {
    "normal": [
        {"name": "mu", "label": "μ (mean)", "type": "float", "default": 0.0},
        {"name": "sigma", "label": "σ (std dev)", "type": "pos", "default": 1.0},
    ],
    "standard_normal": [],
    "student_t": [{"name": "df", "label": "df (degrees of freedom)", "type": "pos", "default": 10.0}],
    "chi_square": [{"name": "df", "label": "df (degrees of freedom)", "type": "pos", "default": 5.0}],
    "f_distribution": [
        {"name": "df1", "label": "df1 (numerator df)", "type": "pos", "default": 5.0},
        {"name": "df2", "label": "df2 (denominator df)", "type": "pos", "default": 10.0},
    ],
    "exponential": [{"name": "rate", "label": "λ (rate)", "type": "pos", "default": 1.0}],
    "continuous_uniform": [
        {"name": "a", "label": "a (lower bound)", "type": "float", "default": 0.0},
        {"name": "b", "label": "b (upper bound)", "type": "float", "default": 1.0},
    ],
    "gamma_dist": [
        {"name": "alpha", "label": "α (shape)", "type": "pos", "default": 2.0},
        {"name": "beta", "label": "β (rate)", "type": "pos", "default": 1.0},
    ],
    "beta_dist": [
        {"name": "a_param", "label": "α (shape 1)", "type": "pos", "default": 2.0},
        {"name": "b_param", "label": "β (shape 2)", "type": "pos", "default": 2.0},
    ],
    "lognormal": [
        {"name": "mu", "label": "μ (log-scale mean)", "type": "float", "default": 0.0},
        {"name": "sigma", "label": "σ (log-scale std dev)", "type": "pos", "default": 1.0},
    ],
    "cauchy": [
        {"name": "x0", "label": "x₀ (location)", "type": "float", "default": 0.0},
        {"name": "gamma", "label": "γ (scale)", "type": "pos", "default": 1.0},
    ],
    "laplace": [
        {"name": "mu", "label": "μ (location)", "type": "float", "default": 0.0},
        {"name": "b_scale", "label": "b (scale)", "type": "pos", "default": 1.0},
    ],
}

DISCRETE_QUERY_TYPES = ["P(X=k)", "P(X<=k)", "P(X<k)", "P(X>k)", "P(X>=k)", "P(a<=X<=b)", "inverse"]
CONTINUOUS_QUERY_TYPES = ["f(x)", "P(X<=a)", "P(X<a)", "P(X>a)", "P(X>=a)", "P(a<=X<=b)", "inverse"]

_law = lambda mod, func, entry_params: {"module": mod, "func": func, "params_spec": entry_params}

NOTATION_SYMBOLS = {
    "bernoulli": "Bernoulli", "binomial": "Bin", "poisson": "Poisson", "geometric": "Geom",
    "negative_binomial": "NB", "hypergeometric": "Hyper", "discrete_uniform": "DUnif", "multinomial": "Multinom",
    "normal": "N", "standard_normal": "N", "student_t": "t", "chi_square": "χ²", "f_distribution": "F",
    "exponential": "Exp", "continuous_uniform": "Unif", "gamma_dist": "Gamma", "beta_dist": "Beta",
    "lognormal": "LogNormal", "cauchy": "Cauchy", "laplace": "Laplace",
}

SUITES = {
    "discrete": {
        "title_en": "Discrete Laws", "title_fr": "Lois Discrètes",
        "desc_en": "Probability calculators for discrete random variables (PMF, CDF, and inverse queries).",
        "desc_fr": "Calculateurs de probabilité pour variables aléatoires discrètes.",
        "icon": "🎲",
        "items": [
            {"id": "bernoulli", "name": "Bernoulli", "name_fr": "Bernoulli", "entry": "D", "module": "laws.discrete.bernoulli", "func": "run_bernoulli_calc", "params_spec": DISCRETE_LAW_PARAMS["bernoulli"], "query_types": DISCRETE_QUERY_TYPES,
             "desc_en": "Single trial with two outcomes (success/failure). Use when modeling one yes/no event.", "desc_fr": "Un seul essai à deux issues (succès/échec). À utiliser pour modéliser un événement oui/non unique."},
            {"id": "binomial", "name": "Binomial", "name_fr": "Binomiale", "entry": "D", "module": "laws.discrete.binomial", "func": "run_binomial_calc", "params_spec": DISCRETE_LAW_PARAMS["binomial"], "query_types": DISCRETE_QUERY_TYPES,
             "desc_en": "Number of successes in n independent Bernoulli trials. Use for repeated yes/no experiments.", "desc_fr": "Nombre de succès dans n essais de Bernoulli indépendants. À utiliser pour des expériences oui/non répétées."},
            {"id": "poisson", "name": "Poisson", "name_fr": "Poisson", "entry": "D", "module": "laws.discrete.poisson", "func": "run_poisson_calc", "params_spec": DISCRETE_LAW_PARAMS["poisson"], "query_types": DISCRETE_QUERY_TYPES,
             "desc_en": "Count of rare events in a fixed interval of time or space. Use for arrival/occurrence counts.", "desc_fr": "Nombre d'événements rares dans un intervalle de temps ou d'espace fixe. À utiliser pour des comptages d'arrivées/occurrences."},
            {"id": "geometric", "name": "Geometric", "name_fr": "Géométrique", "entry": "D", "module": "laws.discrete.geometric", "func": "run_geometric_calc", "params_spec": DISCRETE_LAW_PARAMS["geometric"], "query_types": DISCRETE_QUERY_TYPES,
             "desc_en": "Number of trials until the first success. Use for 'how long until it happens' questions.", "desc_fr": "Nombre d'essais jusqu'au premier succès. À utiliser pour les questions du type « combien de temps avant que cela n'arrive »."},
            {"id": "negative_binomial", "name": "Negative Binomial", "name_fr": "Binomiale Négative", "entry": "D", "module": "laws.discrete.negative_binomial", "func": "run_negative_binomial_calc", "params_spec": DISCRETE_LAW_PARAMS["negative_binomial"], "query_types": DISCRETE_QUERY_TYPES,
             "desc_en": "Number of trials until the r-th success. Generalizes the geometric distribution.", "desc_fr": "Nombre d'essais jusqu'au r-ième succès. Généralise la loi géométrique."},
            {"id": "hypergeometric", "name": "Hypergeometric", "name_fr": "Hypergéométrique", "entry": "D", "module": "laws.discrete.hypergeometric", "func": "run_hypergeometric_calc", "params_spec": DISCRETE_LAW_PARAMS["hypergeometric"], "query_types": DISCRETE_QUERY_TYPES,
             "desc_en": "Successes drawn without replacement from a finite population. Use for sampling-without-replacement problems.", "desc_fr": "Succès tirés sans remise d'une population finie. À utiliser pour les problèmes d'échantillonnage sans remise."},
            {"id": "discrete_uniform", "name": "Discrete Uniform", "name_fr": "Uniforme Discrète", "entry": "D", "module": "laws.discrete.discrete_uniform", "func": "run_discrete_uniform_calc", "params_spec": DISCRETE_LAW_PARAMS["discrete_uniform"], "query_types": DISCRETE_QUERY_TYPES,
             "desc_en": "Every integer outcome in a range is equally likely. Use for dice-like equal-chance outcomes.", "desc_fr": "Chaque résultat entier dans un intervalle est également probable. À utiliser pour des résultats équiprobables comme un dé."},
            {"id": "multinomial", "name": "Multinomial", "name_fr": "Multinomiale", "entry": "D", "module": "laws.discrete.multinomial", "func": "run_multinomial_calc", "params_spec": DISCRETE_LAW_PARAMS["multinomial"], "query_types": ["P(X=k)"],
             "desc_en": "Joint outcome counts across more than two categories. Only the joint PMF query is supported (CDF/inverse are undefined for multinomial).", "desc_fr": "Effectifs conjoints sur plus de deux catégories. Seule la requête de probabilité conjointe est prise en charge (CDF/inverse non définis pour la multinomiale)."},
        ],
    },
    "continuous": {
        "title_en": "Continuous Laws", "title_fr": "Lois Continues",
        "desc_en": "Probability calculators for continuous random variables (PDF, CDF, and inverse queries).",
        "desc_fr": "Calculateurs de probabilité pour variables aléatoires continues.",
        "icon": "📈",
        "items": [
            {"id": "normal", "name": "Normal", "name_fr": "Normale", "entry": "D", "module": "laws.continuous.normal", "func": "run_normal_calc", "params_spec": CONTINUOUS_LAW_PARAMS["normal"], "query_types": CONTINUOUS_QUERY_TYPES,
             "desc_en": "The bell curve. Use for naturally symmetric, continuous measurements (heights, errors, etc.).", "desc_fr": "La courbe en cloche. À utiliser pour des mesures continues naturellement symétriques (tailles, erreurs, etc.)."},
            {"id": "standard_normal", "name": "Standard Normal (Z)", "name_fr": "Normale Centrée Réduite (Z)", "entry": "D", "module": "laws.continuous.standard_normal", "func": "run_standard_normal_calc", "params_spec": CONTINUOUS_LAW_PARAMS["standard_normal"], "query_types": CONTINUOUS_QUERY_TYPES,
             "desc_en": "Normal distribution fixed at mean 0, std dev 1. Use for z-score lookups and critical values.", "desc_fr": "Loi normale fixée à moyenne 0, écart-type 1. À utiliser pour les cotes Z et les valeurs critiques."},
            {"id": "student_t", "name": "Student's t", "name_fr": "Student (t)", "entry": "D", "module": "laws.continuous.student_t", "func": "run_student_t_calc", "params_spec": CONTINUOUS_LAW_PARAMS["student_t"], "query_types": CONTINUOUS_QUERY_TYPES,
             "desc_en": "Heavier-tailed than normal; used when estimating a mean from a small sample with unknown variance.", "desc_fr": "Queues plus épaisses que la normale ; utilisée pour estimer une moyenne à partir d'un petit échantillon à variance inconnue."},
            {"id": "chi_square", "name": "Chi-Square", "name_fr": "Chi-Carré", "entry": "D", "module": "laws.continuous.chi_square", "func": "run_chi_square_calc", "params_spec": CONTINUOUS_LAW_PARAMS["chi_square"], "query_types": CONTINUOUS_QUERY_TYPES,
             "desc_en": "Sum of squared standard normals. Used for variance tests and goodness-of-fit.", "desc_fr": "Somme de carrés de normales centrées réduites. Utilisée pour les tests de variance et d'ajustement."},
            {"id": "f_distribution", "name": "F Distribution", "name_fr": "Loi de Fisher (F)", "entry": "D", "module": "laws.continuous.f_distribution", "func": "run_f_distribution_calc", "params_spec": CONTINUOUS_LAW_PARAMS["f_distribution"], "query_types": CONTINUOUS_QUERY_TYPES,
             "desc_en": "Ratio of two chi-square variables. Used for comparing variances and in ANOVA.", "desc_fr": "Rapport de deux variables chi-carré. Utilisée pour comparer des variances et en ANOVA."},
            {"id": "exponential", "name": "Exponential", "name_fr": "Exponentielle", "entry": "D", "module": "laws.continuous.exponential", "func": "run_exponential_calc", "params_spec": CONTINUOUS_LAW_PARAMS["exponential"], "query_types": CONTINUOUS_QUERY_TYPES,
             "desc_en": "Waiting time between events in a Poisson process. Use for time-to-next-event questions.", "desc_fr": "Temps d'attente entre deux événements d'un processus de Poisson. À utiliser pour les questions de délai avant le prochain événement."},
            {"id": "continuous_uniform", "name": "Continuous Uniform", "name_fr": "Uniforme Continue", "entry": "D", "module": "laws.continuous.continuous_uniform", "func": "run_continuous_uniform_calc", "params_spec": CONTINUOUS_LAW_PARAMS["continuous_uniform"], "query_types": CONTINUOUS_QUERY_TYPES,
             "desc_en": "Every value in an interval is equally likely. Use for flat, equal-chance continuous outcomes.", "desc_fr": "Chaque valeur d'un intervalle est également probable. À utiliser pour des résultats continus équiprobables."},
            {"id": "gamma_dist", "name": "Gamma", "name_fr": "Gamma", "entry": "D", "module": "laws.continuous.gamma_dist", "func": "run_gamma_dist_calc", "params_spec": CONTINUOUS_LAW_PARAMS["gamma_dist"], "query_types": CONTINUOUS_QUERY_TYPES,
             "desc_en": "Sum of exponential waiting times. Use for skewed positive durations/amounts.", "desc_fr": "Somme de temps d'attente exponentiels. À utiliser pour des durées/montants positifs asymétriques."},
            {"id": "beta_dist", "name": "Beta", "name_fr": "Bêta", "entry": "D", "module": "laws.continuous.beta_dist", "func": "run_beta_dist_calc", "params_spec": CONTINUOUS_LAW_PARAMS["beta_dist"], "query_types": CONTINUOUS_QUERY_TYPES,
             "desc_en": "Flexible distribution bounded on [0,1]. Use for modeling proportions or probabilities themselves.", "desc_fr": "Loi flexible bornée sur [0,1]. À utiliser pour modéliser des proportions ou des probabilités elles-mêmes."},
            {"id": "lognormal", "name": "Lognormal", "name_fr": "Log-Normale", "entry": "D", "module": "laws.continuous.lognormal", "func": "run_lognormal_calc", "params_spec": CONTINUOUS_LAW_PARAMS["lognormal"], "query_types": CONTINUOUS_QUERY_TYPES,
             "desc_en": "A variable whose logarithm is normal. Use for right-skewed positive quantities like incomes or sizes.", "desc_fr": "Variable dont le logarithme suit une loi normale. À utiliser pour des quantités positives asymétriques à droite comme les revenus ou tailles."},
            {"id": "cauchy", "name": "Cauchy", "name_fr": "Cauchy", "entry": "D", "module": "laws.continuous.cauchy", "func": "run_cauchy_calc", "params_spec": CONTINUOUS_LAW_PARAMS["cauchy"], "query_types": CONTINUOUS_QUERY_TYPES,
             "desc_en": "Heavy-tailed distribution with undefined mean/variance. Use to illustrate pathological tail behavior.", "desc_fr": "Loi à queues épaisses dont la moyenne/variance sont indéfinies. Illustre un comportement de queue pathologique."},
            {"id": "laplace", "name": "Laplace", "name_fr": "Laplace", "entry": "D", "module": "laws.continuous.laplace", "func": "run_laplace_calc", "params_spec": CONTINUOUS_LAW_PARAMS["laplace"], "query_types": CONTINUOUS_QUERY_TYPES,
             "desc_en": "Two mirrored exponential curves ('double exponential'). Use for sharply-peaked symmetric data.", "desc_fr": "Deux courbes exponentielles en miroir (« double exponentielle »). À utiliser pour des données symétriques à pic marqué."},
        ],
    },
    "descriptive": {
        "title_en": "Descriptive Statistics", "title_fr": "Statistique Descriptive",
        "desc_en": "Summary statistics for discrete, continuous (grouped), and bivariate data.",
        "desc_fr": "Statistiques sommaires pour données discrètes, continues et bivariées.",
        "icon": "📊",
        "items": [
            {"id": "discrete_univariate_stats", "name": "Discrete Univariate Stats", "name_fr": "Statistiques Univariées Discrètes", "entry": "DESC_DISCRETE", "module": "descriptive.discrete_univariate_stats", "func": "compute_discrete_stats",
             "desc_en": "Mean, mode, median, variance, quartiles, skewness, kurtosis, Gini index for distinct values with frequencies.", "desc_fr": "Moyenne, mode, médiane, variance, quartiles, asymétrie, aplatissement, indice de Gini pour des valeurs distinctes avec effectifs."},
            {"id": "continuous_univariate_stats", "name": "Continuous Univariate Stats", "name_fr": "Statistiques Univariées Continues", "entry": "DESC_CONTINUOUS", "module": "descriptive.continuous_univariate_stats", "func": "compute_continuous_stats",
             "desc_en": "Grouped-class statistics: mass frequency, mass median, and all standard univariate measures.", "desc_fr": "Statistiques par classes groupées : fréquence massique, médiane massique, et toutes les mesures univariées standards."},
            {"id": "bivariate_stats", "name": "Bivariate Stats", "name_fr": "Statistiques Bivariées", "entry": "C_DESC", "module": "descriptive.bivariate_stats", "func": "compute_bivariate_stats",
             "desc_en": "Covariance, correlation, inter/intra-variance for two paired variables.", "desc_fr": "Covariance, corrélation, inter/intra-variance pour deux variables appariées."},
        ],
    },
    "parametric": {
        "title_en": "Parametric Tests", "title_fr": "Tests Paramétriques",
        "desc_en": "Classic hypothesis tests assuming a known distributional form (Z, t, F).",
        "desc_fr": "Tests d'hypothèses paramétriques classiques.",
        "icon": "🧪",
        "items": [
            {"id": "z_test_one_sample", "name": "Z-Test (One Sample)", "name_fr": "Test Z (Échantillon unique)", "entry": "A1", "module": "tests.parametric.z_test_one_sample", "func": "run_z_test_one_sample",
             "desc_en": "Test a population mean against a hypothesized value when population std dev is known.", "desc_fr": "Teste une moyenne de population par rapport à une valeur hypothétique lorsque l'écart-type de la population est connu."},
            {"id": "t_test_one_sample", "name": "t-Test (One Sample)", "name_fr": "Test t (Échantillon unique)", "entry": "A1", "module": "tests.parametric.t_test_one_sample", "func": "run_t_test_one_sample",
             "desc_en": "Test a population mean against a hypothesized value when population std dev is unknown.", "desc_fr": "Teste une moyenne de population par rapport à une valeur hypothétique lorsque l'écart-type de la population est inconnu."},
            {"id": "t_test_two_sample", "name": "t-Test (Two Sample)", "name_fr": "Test t (Deux échantillons)", "entry": "B2", "module": "tests.parametric.t_test_two_sample", "func": "run_t_test_two_sample",
             "desc_en": "Compare the means of two independent groups; auto-selects pooled or Welch's t-test via an F-test.", "desc_fr": "Compare les moyennes de deux groupes indépendants ; sélectionne automatiquement le test t regroupé ou de Welch via un test F."},
            {"id": "t_test_paired", "name": "t-Test (Paired)", "name_fr": "Test t (Apparié)", "entry": "B2", "module": "tests.parametric.t_test_paired", "func": "run_t_test_paired",
             "desc_en": "Compare means of two related/matched samples (before-after, matched pairs).", "desc_fr": "Compare les moyennes de deux échantillons appariés/liés (avant-après, paires assorties)."},
            {"id": "f_test_variance", "name": "F-Test (Variance Equality)", "name_fr": "Test F (Égalité des variances)", "entry": "B2", "module": "tests.parametric.f_test_variance", "func": "run_f_test_variance",
             "desc_en": "Test whether two populations have equal variance.", "desc_fr": "Teste si deux populations ont une variance égale."},
            {"id": "z_test_one_proportion", "name": "Z-Test (One Proportion)", "name_fr": "Test Z (Une proportion)", "entry": "PROP1", "module": "tests.parametric.z_test_one_proportion", "func": "run_z_test_one_proportion",
             "desc_en": "Test a population proportion against a hypothesized value.", "desc_fr": "Teste une proportion de population par rapport à une valeur hypothétique."},
            {"id": "z_test_two_proportion", "name": "Z-Test (Two Proportions)", "name_fr": "Test Z (Deux proportions)", "entry": "PROP2", "module": "tests.parametric.z_test_two_proportion", "func": "run_z_test_two_proportion",
             "desc_en": "Compare proportions between two independent groups.", "desc_fr": "Compare les proportions entre deux groupes indépendants."},
            {"id": "confidence_interval", "name": "Confidence Interval Calculator", "name_fr": "Calculateur d'Intervalle de Confiance", "entry": "CI", "module": "tests.parametric.confidence_interval", "func": "run_confidence_interval",
             "desc_en": "Build a confidence interval for a mean, proportion, or their two-sample differences.", "desc_fr": "Construit un intervalle de confiance pour une moyenne, une proportion, ou leurs différences à deux échantillons."},
            {"id": "sample_size_calculator", "name": "Sample Size Calculator", "name_fr": "Calculateur de Taille d'Échantillon", "entry": "SS", "module": "tests.parametric.sample_size_calculator", "func": "run_sample_size_calculator",
             "desc_en": "Determine the sample size needed for a target margin of error, confidence, or power.", "desc_fr": "Détermine la taille d'échantillon nécessaire pour une marge d'erreur, une confiance ou une puissance cible."},
        ],
    },
    "nonparametric": {
        "title_en": "Non-Parametric Tests", "title_fr": "Tests Non-Paramétriques",
        "desc_en": "Distribution-free hypothesis tests based on ranks or signs.",
        "desc_fr": "Tests d'hypothèses non-paramétriques.",
        "icon": "🔀",
        "items": [
            {"id": "mann_whitney", "name": "Mann-Whitney U", "name_fr": "U de Mann-Whitney", "entry": "B2", "module": "tests.nonparametric.mann_whitney", "func": "run_mann_whitney",
             "desc_en": "Rank-based alternative to the two-sample t-test when normality is doubtful.", "desc_fr": "Alternative basée sur les rangs au test t à deux échantillons lorsque la normalité est douteuse."},
            {"id": "wilcoxon_signed_rank", "name": "Wilcoxon Signed-Rank", "name_fr": "Wilcoxon (Rangs Signés)", "entry": "B2", "module": "tests.nonparametric.wilcoxon_signed_rank", "func": "run_wilcoxon_signed_rank",
             "desc_en": "Rank-based alternative to the paired t-test for matched samples.", "desc_fr": "Alternative basée sur les rangs au test t apparié pour échantillons appariés."},
            {"id": "kruskal_wallis", "name": "Kruskal-Wallis H", "name_fr": "H de Kruskal-Wallis", "entry": "BN", "module": "tests.nonparametric.kruskal_wallis", "func": "run_kruskal_wallis",
             "desc_en": "Rank-based alternative to one-way ANOVA for 3 or more independent groups.", "desc_fr": "Alternative basée sur les rangs à l'ANOVA à un facteur pour 3 groupes indépendants ou plus."},
            {"id": "sign_test", "name": "Sign Test", "name_fr": "Test du Signe", "entry": "B2", "module": "tests.nonparametric.sign_test", "func": "run_sign_test",
             "desc_en": "Simple test of the median difference using only the signs of paired differences.", "desc_fr": "Test simple de la différence médiane utilisant uniquement les signes des différences appariées."},
            {"id": "runs_test", "name": "Runs Test", "name_fr": "Test des Suites (Runs)", "entry": "RUNS", "module": "tests.nonparametric.runs_test", "func": "run_runs_test",
             "desc_en": "Tests whether a sequence of data is random (no serial pattern).", "desc_fr": "Teste si une séquence de données est aléatoire (absence de motif sériel)."},
        ],
    },
    "anova": {
        "title_en": "ANOVA", "title_fr": "ANOVA",
        "desc_en": "Analysis of variance across two or more groups, plus supporting variance-homogeneity tests.",
        "desc_fr": "Analyse de variance entre deux groupes ou plus.",
        "icon": "📐",
        "items": [
            {"id": "anova_one_way", "name": "One-Way ANOVA", "name_fr": "ANOVA à un facteur", "entry": "BN", "module": "tests.anova.anova_one_way", "func": "run_anova_one_way",
             "desc_en": "Compare means across 3+ independent groups on one factor, with Tukey HSD post-hoc if significant.", "desc_fr": "Compare les moyennes entre 3 groupes indépendants ou plus sur un facteur, avec test post-hoc de Tukey si significatif."},
            {"id": "anova_two_way_no_replication", "name": "Two-Way ANOVA (No Replication)", "name_fr": "ANOVA à deux facteurs (Sans répétition)", "entry": "MATRIX", "module": "tests.anova.anova_two_way_no_replication", "func": "run_anova_two_way_no_replication",
             "desc_en": "Test two factors simultaneously with exactly one observation per cell (additive model).", "desc_fr": "Teste deux facteurs simultanément avec exactement une observation par cellule (modèle additif)."},
            {"id": "anova_two_way_replication", "name": "Two-Way ANOVA (With Replication)", "name_fr": "ANOVA à deux facteurs (Avec répétition)", "entry": "TWOWAY_REP", "module": "tests.anova.anova_two_way_replication", "func": "run_anova_two_way_replication",
             "desc_en": "Test two factors and their interaction with multiple observations per cell.", "desc_fr": "Teste deux facteurs et leur interaction avec plusieurs observations par cellule."},
            {"id": "bartlett_test", "name": "Bartlett's Test", "name_fr": "Test de Bartlett", "entry": "BN", "module": "tests.anova.bartlett_test", "func": "run_bartlett_test",
             "desc_en": "Test equality of variances across groups (sensitive to non-normality).", "desc_fr": "Teste l'égalité des variances entre groupes (sensible à la non-normalité)."},
            {"id": "levene_test", "name": "Levene's Test", "name_fr": "Test de Levene", "entry": "BN", "module": "tests.anova.levene_test", "func": "run_levene_test",
             "desc_en": "Robust test of equality of variances across groups, less sensitive to non-normality.", "desc_fr": "Test robuste de l'égalité des variances entre groupes, moins sensible à la non-normalité."},
            {"id": "brown_forsythe_test", "name": "Brown-Forsythe Test", "name_fr": "Test de Brown-Forsythe", "entry": "BN", "module": "tests.anova.brown_forsythe_test", "func": "run_brown_forsythe_test",
             "desc_en": "Median-centered variant of Levene's test, robust test of equality of variances even under skewed distributions.", "desc_fr": "Variante du test de Levene centrée sur la médiane, robuste à l'égalité des variances même sous des distributions asymétriques."},
        ],
    },
    "correlation": {
        "title_en": "Correlation", "title_fr": "Corrélation",
        "desc_en": "Association strength and significance between two or more variables.",
        "desc_fr": "Force et significativité de l'association entre variables.",
        "icon": "🔗",
        "items": [
            {"id": "pearson_correlation", "name": "Pearson Correlation", "name_fr": "Corrélation de Pearson", "entry": "C", "module": "tests.correlation.pearson_correlation", "func": "run_pearson_correlation",
             "desc_en": "Linear correlation between two continuous variables, with significance test and Fisher Z CI.", "desc_fr": "Corrélation linéaire entre deux variables continues, avec test de significativité et IC de Fisher Z."},
            {"id": "kendall_tau", "name": "Kendall's Tau", "name_fr": "Tau de Kendall", "entry": "C", "module": "tests.correlation.kendall_tau", "func": "run_kendall_tau",
             "desc_en": "Rank-based concordance measure of association, robust to outliers and ties.", "desc_fr": "Mesure de concordance basée sur les rangs, robuste aux valeurs aberrantes et aux ex-aequo."},
            {"id": "spearman_correlation", "name": "Spearman Correlation", "name_fr": "Corrélation de Spearman", "entry": "C", "module": "tests.correlation.spearman_correlation", "func": "run_spearman_correlation",
             "desc_en": "Rank-based monotonic correlation between two variables.", "desc_fr": "Corrélation monotone basée sur les rangs entre deux variables."},
            {"id": "correlation_matrix", "name": "Correlation Matrix", "name_fr": "Matrice de Corrélation", "entry": "MATRIX_VARS", "module": "tests.correlation.correlation_matrix", "func": "run_correlation_matrix",
             "desc_en": "Full pairwise Pearson correlation matrix across many variables with a heatmap and PCA-style eigen-decomposition.", "desc_fr": "Matrice complète de corrélation de Pearson par paires sur plusieurs variables, avec carte de chaleur et décomposition en valeurs propres de type ACP."},
        ],
    },
    "regression": {
        "title_en": "Regression", "title_fr": "Régression",
        "desc_en": "Model a response variable as a function of one or more predictors.",
        "desc_fr": "Modéliser une variable de réponse en fonction de prédicteurs.",
        "icon": "📉",
        "items": [
            {"id": "simple_linear_regression", "name": "Simple Linear Regression", "name_fr": "Régression Linéaire Simple", "entry": "SLR", "module": "tests.regression.simple_linear_regression", "func": "run_simple_linear_regression",
             "desc_en": "Fit a straight line Y = a + bX to bivariate data, with prediction and diagnostic plots.", "desc_fr": "Ajuste une droite Y = a + bX à des données bivariées, avec prédiction et graphiques diagnostiques."},
            {"id": "multiple_linear_regression", "name": "Multiple Linear Regression", "name_fr": "Régression Linéaire Multiple", "entry": "MULTI_REG", "module": "tests.regression.multiple_linear_regression", "func": "run_multiple_linear_regression",
             "desc_en": "Fit a linear model with two or more predictors, with coefficient significance and R².", "desc_fr": "Ajuste un modèle linéaire avec deux prédicteurs ou plus, avec significativité des coefficients et R²."},
            {"id": "polynomial_regression", "name": "Polynomial Regression", "name_fr": "Régression Polynomiale", "entry": "POLY_REG", "module": "tests.regression.polynomial_regression", "func": "run_polynomial_regression",
             "desc_en": "Fit a curved (degree 1-5) polynomial model to bivariate data.", "desc_fr": "Ajuste un modèle polynomial courbe (degré 1 à 5) à des données bivariées."},
        ],
    },
    "gof": {
        "title_en": "Goodness of Fit", "title_fr": "Ajustement",
        "desc_en": "Chi-square tests of whether sample data follows a hypothesized distribution.",
        "desc_fr": "Tests du chi-carré d'ajustement à une loi théorique.",
        "icon": "✅",
        "items": [
            {"id": "gof_bernoulli", "name": "GOF: Bernoulli", "name_fr": "Ajustement : Bernoulli", "entry": "GOF_DISCRETE", "module": "tests.goodness_of_fit.gof_bernoulli", "func": "run_gof_bernoulli", "law_params": [{"name": "p0", "label": "p₀ (hypothesized)", "type": "prob", "default": 0.5}],
             "desc_en": "Test whether binary sample data follows a Bernoulli(p₀) distribution.", "desc_fr": "Teste si des données binaires suivent une loi de Bernoulli(p₀)."},
            {"id": "gof_binomial", "name": "GOF: Binomial", "name_fr": "Ajustement : Binomiale", "entry": "GOF_DISCRETE", "module": "tests.goodness_of_fit.gof_binomial", "func": "run_gof_binomial", "law_params": [{"name": "n", "label": "n (trials)", "type": "posint", "default": 10}],
             "desc_en": "Test whether count data follows a Binomial(n, p̂) distribution.", "desc_fr": "Teste si des données de comptage suivent une loi Binomiale(n, p̂)."},
            {"id": "gof_poisson", "name": "GOF: Poisson", "name_fr": "Ajustement : Poisson", "entry": "GOF_DISCRETE", "module": "tests.goodness_of_fit.gof_poisson", "func": "run_gof_poisson", "law_params": [],
             "desc_en": "Test whether count data follows a Poisson distribution.", "desc_fr": "Teste si des données de comptage suivent une loi de Poisson."},
            {"id": "gof_geometric", "name": "GOF: Geometric", "name_fr": "Ajustement : Géométrique", "entry": "GOF_DISCRETE", "module": "tests.goodness_of_fit.gof_geometric", "func": "run_gof_geometric", "law_params": [],
             "desc_en": "Test whether trial-count data follows a Geometric distribution.", "desc_fr": "Teste si des données de nombre d'essais suivent une loi Géométrique."},
            {"id": "gof_negative_binomial", "name": "GOF: Negative Binomial", "name_fr": "Ajustement : Binomiale Négative", "entry": "GOF_DISCRETE", "module": "tests.goodness_of_fit.gof_negative_binomial", "func": "run_gof_negative_binomial", "law_params": [{"name": "r", "label": "r (successes)", "type": "posint", "default": 3}],
             "desc_en": "Test whether count data follows a Negative Binomial(r, p̂) distribution.", "desc_fr": "Teste si des données de comptage suivent une loi Binomiale Négative(r, p̂)."},
            {"id": "gof_hypergeometric", "name": "GOF: Hypergeometric", "name_fr": "Ajustement : Hypergéométrique", "entry": "GOF_DISCRETE", "module": "tests.goodness_of_fit.gof_hypergeometric", "func": "run_gof_hypergeometric", "law_params": [{"name": "M", "label": "M (population size)", "type": "posint", "default": 50}, {"name": "n", "label": "n (success states)", "type": "posint", "default": 20}, {"name": "N_sample", "label": "N (sample size)", "type": "posint", "default": 10}],
             "desc_en": "Test whether sampling-without-replacement data follows a Hypergeometric distribution.", "desc_fr": "Teste si des données d'échantillonnage sans remise suivent une loi Hypergéométrique."},
            {"id": "gof_discrete_uniform", "name": "GOF: Discrete Uniform", "name_fr": "Ajustement : Uniforme Discrète", "entry": "GOF_DISCRETE", "module": "tests.goodness_of_fit.gof_discrete_uniform", "func": "run_gof_discrete_uniform", "law_params": [{"name": "a", "label": "a (lower bound)", "type": "int", "default": 1}, {"name": "b", "label": "b (upper bound)", "type": "int", "default": 6}],
             "desc_en": "Test whether integer data is uniformly distributed over [a, b].", "desc_fr": "Teste si des données entières sont uniformément distribuées sur [a, b]."},
            {"id": "gof_multinomial", "name": "GOF: Multinomial", "name_fr": "Ajustement : Multinomiale", "entry": "GOF_MULTINOMIAL", "module": "tests.goodness_of_fit.gof_multinomial", "func": "run_gof_multinomial",
             "desc_en": "Test whether observed category counts match hypothesized category probabilities.", "desc_fr": "Teste si les effectifs de catégories observés correspondent aux probabilités de catégories hypothétiques."},
            {"id": "gof_normal", "name": "GOF: Normal", "name_fr": "Ajustement : Normale", "entry": "GOF_CONTINUOUS", "module": "tests.goodness_of_fit.gof_normal", "func": "run_gof_normal", "law_params": [],
             "desc_en": "Test whether continuous data follows a Normal distribution (parameters estimated from data).", "desc_fr": "Teste si des données continues suivent une loi Normale (paramètres estimés à partir des données)."},
            {"id": "gof_standard_normal", "name": "GOF: Standard Normal", "name_fr": "Ajustement : Normale Centrée Réduite", "entry": "GOF_CONTINUOUS", "module": "tests.goodness_of_fit.gof_standard_normal", "func": "run_gof_standard_normal", "law_params": [],
             "desc_en": "Test whether continuous data follows N(0,1) exactly (parameters fixed, not estimated).", "desc_fr": "Teste si des données continues suivent exactement N(0,1) (paramètres fixés, non estimés)."},
            {"id": "gof_student_t", "name": "GOF: Student's t", "name_fr": "Ajustement : Student (t)", "entry": "GOF_CONTINUOUS", "module": "tests.goodness_of_fit.gof_student_t", "func": "run_gof_student_t", "law_params": [{"name": "df", "label": "df (degrees of freedom)", "type": "pos", "default": 10.0}],
             "desc_en": "Test whether continuous data follows a Student's t distribution with given df.", "desc_fr": "Teste si des données continues suivent une loi de Student avec un df donné."},
            {"id": "gof_chi_square", "name": "GOF: Chi-Square", "name_fr": "Ajustement : Chi-Carré", "entry": "GOF_CONTINUOUS", "module": "tests.goodness_of_fit.gof_chi_square", "func": "run_gof_chi_square", "law_params": [{"name": "df", "label": "df (degrees of freedom)", "type": "pos", "default": 5.0}],
             "desc_en": "Test whether continuous data follows a Chi-Square distribution with given df.", "desc_fr": "Teste si des données continues suivent une loi du Chi-Carré avec un df donné."},
            {"id": "gof_f_distribution", "name": "GOF: F Distribution", "name_fr": "Ajustement : Loi de Fisher (F)", "entry": "GOF_CONTINUOUS", "module": "tests.goodness_of_fit.gof_f_distribution", "func": "run_gof_f_distribution", "law_params": [{"name": "df1", "label": "df1", "type": "pos", "default": 5.0}, {"name": "df2", "label": "df2", "type": "pos", "default": 10.0}],
             "desc_en": "Test whether continuous data follows an F distribution with given df1, df2.", "desc_fr": "Teste si des données continues suivent une loi F avec df1, df2 donnés."},
            {"id": "gof_exponential", "name": "GOF: Exponential", "name_fr": "Ajustement : Exponentielle", "entry": "GOF_CONTINUOUS", "module": "tests.goodness_of_fit.gof_exponential", "func": "run_gof_exponential", "law_params": [],
             "desc_en": "Test whether continuous waiting-time data follows an Exponential distribution.", "desc_fr": "Teste si des données continues de temps d'attente suivent une loi Exponentielle."},
            {"id": "gof_continuous_uniform", "name": "GOF: Continuous Uniform", "name_fr": "Ajustement : Uniforme Continue", "entry": "GOF_CONTINUOUS", "module": "tests.goodness_of_fit.gof_continuous_uniform", "func": "run_gof_continuous_uniform", "law_params": [{"name": "a", "label": "a (lower bound)", "type": "float", "default": 0.0}, {"name": "b", "label": "b (upper bound)", "type": "float", "default": 1.0}],
             "desc_en": "Test whether continuous data is uniformly distributed over [a, b].", "desc_fr": "Teste si des données continues sont uniformément distribuées sur [a, b]."},
            {"id": "gof_gamma", "name": "GOF: Gamma", "name_fr": "Ajustement : Gamma", "entry": "GOF_CONTINUOUS", "module": "tests.goodness_of_fit.gof_gamma", "func": "run_gof_gamma", "law_params": [],
             "desc_en": "Test whether continuous data follows a Gamma distribution.", "desc_fr": "Teste si des données continues suivent une loi Gamma."},
            {"id": "gof_beta", "name": "GOF: Beta", "name_fr": "Ajustement : Bêta", "entry": "GOF_CONTINUOUS", "module": "tests.goodness_of_fit.gof_beta", "func": "run_gof_beta", "law_params": [],
             "desc_en": "Test whether [0,1]-bounded data follows a Beta distribution.", "desc_fr": "Teste si des données bornées sur [0,1] suivent une loi Bêta."},
            {"id": "gof_lognormal", "name": "GOF: Lognormal", "name_fr": "Ajustement : Log-Normale", "entry": "GOF_CONTINUOUS", "module": "tests.goodness_of_fit.gof_lognormal", "func": "run_gof_lognormal", "law_params": [],
             "desc_en": "Test whether positive continuous data follows a Lognormal distribution.", "desc_fr": "Teste si des données continues positives suivent une loi Log-Normale."},
            {"id": "gof_cauchy", "name": "GOF: Cauchy", "name_fr": "Ajustement : Cauchy", "entry": "GOF_CONTINUOUS", "module": "tests.goodness_of_fit.gof_cauchy", "func": "run_gof_cauchy", "law_params": [],
             "desc_en": "Test whether continuous data follows a Cauchy distribution.", "desc_fr": "Teste si des données continues suivent une loi de Cauchy."},
            {"id": "gof_laplace", "name": "GOF: Laplace", "name_fr": "Ajustement : Laplace", "entry": "GOF_CONTINUOUS", "module": "tests.goodness_of_fit.gof_laplace", "func": "run_gof_laplace", "law_params": [],
             "desc_en": "Test whether continuous data follows a Laplace distribution.", "desc_fr": "Teste si des données continues suivent une loi de Laplace."},
            {"id": "chi_sq_independence", "name": "Chi-Square Independence", "name_fr": "Indépendance du Chi-Carré", "entry": "CONTINGENCY", "module": "tests.goodness_of_fit.chi_sq_independence", "func": "run_chi_sq_independence",
             "desc_en": "Test whether two categorical variables are independent using a contingency table.", "desc_fr": "Teste si deux variables catégorielles sont indépendantes à l'aide d'un tableau de contingence."},
            {"id": "chi_sq_homogeneity", "name": "Chi-Square Homogeneity", "name_fr": "Homogénéité du Chi-Carré", "entry": "CONTINGENCY", "module": "tests.goodness_of_fit.chi_sq_homogeneity", "func": "run_chi_sq_homogeneity",
             "desc_en": "Test whether several populations/samples share the same distribution across the categories of one variable.", "desc_fr": "Teste si plusieurs populations/échantillons partagent la même distribution sur les catégories d'une variable."},
        ],
    },
}


def get_all_items_flat():
    """Returns a flat list of (suite_key, item) tuples for global search."""
    flat = []
    for suite_key, suite in SUITES.items():
        for item in suite["items"]:
            flat.append((suite_key, item))
    return flat


def get_item(suite_key, item_id):
    for item in SUITES[suite_key]["items"]:
        if item["id"] == item_id:
            return item
    return None
