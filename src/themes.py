"""
Emerging-industry / speculative-theme knowledge base.

WHY THIS FILE EXISTS
--------------------
The Speculative Score (src/speculative.py) asks a forward-looking question:
"How much potential does this stock have to EXPLODE in value if its bullish
thesis or emerging-industry opportunity plays out?"

That question cannot be answered from financial statements alone. A
company's income statement tells you what it earned last year; it tells you
nothing about whether it sits in front of a $10B market or a $1T market.
Answering the question honestly requires an estimate of the FUTURE market
opportunity -- which is external, forward-looking information.

Rather than invent those numbers inside the scoring code (which would make
them invisible and unauditable), every forward-looking industry assumption
this model uses is centralized here, in ONE inspectable, editable file, with
an explicit source note on each theme. The scoring code in
src/speculative.py contains no hard-coded company or industry constants of
its own -- it reads everything from this table.

HONESTY ABOUT WHAT THESE NUMBERS ARE
------------------------------------
The `tam_usd` and `industry_cagr` figures below are approximate
mid-to-late-2030s market-size estimates drawn from widely published
market-research and consulting forecasts (McKinsey, BCG, Precedence/Grand
View/MarketsandMarkets-style industry reports, and company investor-day
disclosures). They are:

  * ESTIMATES, not facts. Different research houses publish figures that
    differ by 2-5x for the same emerging industry. Where sources disagreed,
    a mid-range figure was chosen deliberately rather than the largest
    available headline number.
  * ORDER-OF-MAGNITUDE inputs, not precise forecasts. The scoring code
    consumes them on a LOG scale precisely because the difference between a
    $50B market and a $500B market is meaningful, while the difference
    between $90B and $110B is noise.
  * MODEL ASSUMPTIONS, disclosed as such everywhere they surface in the UI.

The right way to use this file is to treat it as a set of dials you can
audit and adjust. If you disagree that quantum computing reaches ~$90B by
the mid-2030s, change the number here and every score updates consistently.

ANTI-HYPE DESIGN
----------------
Membership in a theme is NOT sufficient to earn points. `keywords` matching
a business description is the weakest form of evidence and is deliberately
gated in src/speculative.py by `theme_evidence_strength()`, which requires
corroborating financial evidence (real R&D spend, real revenue growth, real
gross margin, real capital investment) before a theme's optionality points
are awarded in full. A shell company that merely puts "quantum" or "AI" in
its business summary, with no R&D and no growth, receives a fraction of the
theme's raw potential. See `SpeculativeTheme.breakthrough_potential` and the
gate applied to it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class SpeculativeTheme:
    """One emerging (or mature) industry opportunity profile.

    Attributes
    ----------
    key / name
        Identifier and human-readable label.
    tam_usd
        Estimated total addressable market, in USD, at roughly a 2033-2035
        horizon. See the module docstring for what this figure is and is
        not.
    industry_cagr
        Estimated annualized industry growth rate to that horizon.
    commercial_maturity
        0-1. How commercially PROVEN the industry is today. 1.0 = a
        settled, mature industry where the business model is fully
        established (packaged food, utilities). 0.05 = pre-commercial,
        where essentially no one is yet generating meaningful profit
        (fusion). Low maturity means MORE optionality but LESS certainty --
        the scoring code uses it for the former and handles the latter
        separately in the "Probability of Becoming a Major Winner"
        category.
    breakthrough_potential
        0-1. How much a genuine technological breakthrough could re-rate
        the entire industry's value. High for fields where a working
        solution unlocks an entirely new market (fusion, quantum, gene
        editing); low for industries where progress is incremental.
    catalyst_intensity
        0-1. How catalyst-dense this industry is -- i.e. how often
        discrete, dateable events (approvals, launches, contract awards,
        milestones) cause large repricings for its companies. Biotech
        (binary trial readouts) is high; consumer staples is low.
    winner_take_most
        0-1. How concentrated the eventual economics are. 1.0 = one or two
        firms capture most of the industry's profit (platform/network
        businesses); 0.2 = fragmented and commoditized.
    plausible_winner_share
        Fraction of `tam_usd` in REVENUE that a major eventual winner in
        this industry could plausibly capture. Deliberately conservative:
        this is a "successful outcome," not a monopoly assumption.
    winner_ev_sales
        The EV/Sales multiple a successful company in this industry might
        command AT that future horizon -- i.e. still-growing but no longer
        pre-revenue. Used to convert a potential future revenue figure into
        a potential future market capitalization.
    catalyst_archetypes
        The KINDS of events that re-rate companies in this industry. These
        are industry-level archetypes, presented in the UI as "catalyst
        types that apply to this industry" -- never as claims that a
        specific company has a specific event scheduled. This tool has no
        access to company calendars and never fabricates one.
    keywords
        Lowercase substrings matched against the company's business
        summary / industry / name. Weakest evidence tier; gated (see the
        module docstring).
    tickers
        Explicit, high-confidence membership. Strongest evidence tier.
    source_note
        Where the TAM/CAGR estimate comes from, in plain language.
    """

    key: str
    name: str
    tam_usd: float
    industry_cagr: float
    commercial_maturity: float
    breakthrough_potential: float
    catalyst_intensity: float
    winner_take_most: float
    plausible_winner_share: float
    winner_ev_sales: float
    source_note: str
    catalyst_archetypes: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()
    tickers: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# The theme table.
#
# Ordered roughly from most to least speculative. Add, edit, or remove
# entries freely -- src/speculative.py reads this table and contains no
# industry constants of its own.
# ---------------------------------------------------------------------------
THEMES: tuple[SpeculativeTheme, ...] = (
    SpeculativeTheme(
        key="fusion",
        name="Fusion Energy",
        tam_usd=40e9,
        industry_cagr=0.55,
        commercial_maturity=0.03,
        breakthrough_potential=1.00,
        catalyst_intensity=0.85,
        winner_take_most=0.90,
        plausible_winner_share=0.30,
        winner_ev_sales=14.0,
        source_note="Pre-commercial. TAM reflects an early-2030s first-plant "
                    "market only; the theoretical end-state energy market is "
                    "orders of magnitude larger but is not creditable at this horizon.",
        catalyst_archetypes=("net-energy-gain milestones", "pilot plant construction",
                             "utility power-purchase agreements", "government/defense funding awards"),
        keywords=("fusion energy", "nuclear fusion", "tokamak", "inertial confinement"),
        tickers=(),
    ),
    SpeculativeTheme(
        key="quantum_computing",
        name="Quantum Computing",
        tam_usd=90e9,
        industry_cagr=0.35,
        commercial_maturity=0.12,
        breakthrough_potential=0.98,
        catalyst_intensity=0.80,
        winner_take_most=0.85,
        plausible_winner_share=0.22,
        winner_ev_sales=15.0,
        source_note="Mid-range of published mid-2030s quantum-computing market "
                    "forecasts (~$50B-$130B); McKinsey-style estimates put the "
                    "value-at-stake materially higher but over a longer horizon.",
        catalyst_archetypes=("logical-qubit / error-correction milestones",
                             "quantum advantage demonstrations",
                             "government and national-lab contract awards",
                             "enterprise and cloud-provider partnerships",
                             "commercial system sales and bookings growth"),
        keywords=("quantum computing", "quantum computer", "qubit", "trapped ion",
                  "quantum information", "quantum networking"),
        tickers=("IONQ", "RGTI", "QBTS", "QUBT", "ARQQ"),
    ),
    SpeculativeTheme(
        key="humanoid_robotics",
        name="Humanoid & General-Purpose Robotics",
        tam_usd=180e9,
        industry_cagr=0.45,
        commercial_maturity=0.08,
        breakthrough_potential=0.95,
        catalyst_intensity=0.70,
        winner_take_most=0.80,
        plausible_winner_share=0.22,
        winner_ev_sales=12.0,
        source_note="Early-stage category; estimate reflects mid-2030s forecasts "
                    "for general-purpose/humanoid robots specifically, not the "
                    "much larger established industrial-automation market.",
        catalyst_archetypes=("production-line deployment milestones", "unit-cost reduction",
                             "commercial pilot conversions", "manufacturing capacity ramps"),
        keywords=("humanoid robot", "general purpose robot", "general-purpose robot"),
        tickers=(),
    ),
    SpeculativeTheme(
        key="evtol",
        name="eVTOL & Advanced Air Mobility",
        tam_usd=75e9,
        industry_cagr=0.45,
        commercial_maturity=0.06,
        breakthrough_potential=0.85,
        catalyst_intensity=0.90,
        winner_take_most=0.70,
        plausible_winner_share=0.25,
        winner_ev_sales=10.0,
        source_note="Mid-2030s advanced-air-mobility forecasts. Highly "
                    "regulatory-gated; certification is the dominant variable.",
        catalyst_archetypes=("FAA/EASA type certification stages", "first commercial route launch",
                             "fleet orders from airlines or operators", "manufacturing rate milestones"),
        keywords=("evtol", "electric vertical takeoff", "advanced air mobility", "air taxi"),
        tickers=("JOBY", "ACHR", "EH", "LILM"),
    ),
    SpeculativeTheme(
        key="gene_editing",
        name="Gene Editing & Genetic Medicine",
        tam_usd=55e9,
        industry_cagr=0.25,
        commercial_maturity=0.15,
        breakthrough_potential=0.95,
        catalyst_intensity=0.95,
        winner_take_most=0.65,
        plausible_winner_share=0.20,
        winner_ev_sales=11.0,
        source_note="Mid-2030s gene-editing/genetic-medicine therapeutic market "
                    "estimates. Excludes the broader biopharma market.",
        catalyst_archetypes=("clinical trial readouts (Phase 1/2/3)", "FDA/EMA approval decisions",
                             "pharma partnership and licensing deals", "first commercial launch and payer coverage"),
        keywords=("gene editing", "crispr", "gene therapy", "genetic medicine", "base editing"),
        tickers=("CRSP", "NTLA", "BEAM", "EDIT", "VERV"),
    ),
    SpeculativeTheme(
        key="nuclear_smr",
        name="Advanced Nuclear & Small Modular Reactors",
        tam_usd=280e9,
        industry_cagr=0.24,
        commercial_maturity=0.14,
        breakthrough_potential=0.75,
        catalyst_intensity=0.85,
        winner_take_most=0.60,
        plausible_winner_share=0.18,
        winner_ev_sales=8.0,
        source_note="Mid-2030s SMR/advanced-nuclear build-out forecasts, boosted "
                    "in recent estimates by data-center power demand.",
        catalyst_archetypes=("NRC design certification and licensing milestones",
                             "first-of-a-kind plant construction start",
                             "utility and hyperscaler power agreements",
                             "government loan guarantees and DOE awards"),
        keywords=("small modular reactor", "advanced nuclear", "nuclear reactor",
                  "microreactor", "nuclear fuel"),
        tickers=("OKLO", "SMR", "NNE", "LEU", "LTBR"),
    ),
    SpeculativeTheme(
        key="autonomous_vehicles",
        name="Autonomous Vehicles & Self-Driving Systems",
        tam_usd=650e9,
        industry_cagr=0.30,
        commercial_maturity=0.18,
        breakthrough_potential=0.90,
        catalyst_intensity=0.70,
        winner_take_most=0.85,
        plausible_winner_share=0.20,
        winner_ev_sales=9.0,
        source_note="Mid-2030s robotaxi + autonomous-driving-software forecasts. "
                    "Wide dispersion across sources; a mid-range figure is used.",
        catalyst_archetypes=("driverless permit approvals and geographic expansion",
                             "removal of safety drivers", "OEM licensing deals",
                             "per-mile cost crossover vs. human drivers"),
        keywords=("autonomous vehicle", "self-driving", "robotaxi", "autonomous driving",
                  "driverless", "lidar", "advanced driver assistance"),
        tickers=("LAZR", "OUST", "INVZ", "AUR", "MBLY", "PONY", "WRD"),
    ),
    SpeculativeTheme(
        key="space",
        name="Space Launch, Satellites & Space Economy",
        tam_usd=750e9,
        industry_cagr=0.14,
        commercial_maturity=0.35,
        breakthrough_potential=0.75,
        catalyst_intensity=0.80,
        winner_take_most=0.70,
        plausible_winner_share=0.15,
        winner_ev_sales=8.0,
        source_note="Mid-2030s total space-economy estimates (launch, satellite "
                    "services, earth observation, defense space).",
        catalyst_archetypes=("new launch vehicle first flights and cadence ramps",
                             "constellation deployment milestones",
                             "national-security and NASA contract awards",
                             "commercial service commencement"),
        keywords=("space launch", "satellite", "spacecraft", "launch vehicle",
                  "earth observation", "space station", "orbital"),
        tickers=("RKLB", "ASTS", "PL", "RDW", "LUNR", "SPCE", "BKSY"),
    ),
    SpeculativeTheme(
        key="hydrogen",
        name="Hydrogen & Clean Fuels",
        tam_usd=260e9,
        industry_cagr=0.28,
        commercial_maturity=0.15,
        breakthrough_potential=0.70,
        catalyst_intensity=0.65,
        winner_take_most=0.45,
        plausible_winner_share=0.15,
        winner_ev_sales=5.0,
        source_note="Mid-2030s green-hydrogen production and fuel-cell forecasts. "
                    "Heavily dependent on subsidy regimes.",
        catalyst_archetypes=("electrolyzer cost-per-kilogram milestones",
                             "government subsidy and tax-credit rulings",
                             "large offtake agreements", "gigafactory commissioning"),
        keywords=("hydrogen", "fuel cell", "electrolyzer", "green hydrogen"),
        tickers=("PLUG", "BE", "BLDP", "FCEL"),
    ),
    SpeculativeTheme(
        key="synthetic_bio",
        name="Synthetic Biology & Biomanufacturing",
        tam_usd=110e9,
        industry_cagr=0.24,
        commercial_maturity=0.18,
        breakthrough_potential=0.85,
        catalyst_intensity=0.60,
        winner_take_most=0.55,
        plausible_winner_share=0.18,
        winner_ev_sales=8.0,
        source_note="Mid-2030s synthetic-biology / engineered-organism market estimates.",
        catalyst_archetypes=("commercial-scale fermentation milestones",
                             "major partnership and program expansions",
                             "regulatory clearance for engineered products"),
        keywords=("synthetic biology", "biomanufacturing", "engineered organism",
                  "cell programming"),
        tickers=("DNA", "AMRS", "TWST"),
    ),
    SpeculativeTheme(
        key="ar_vr_spatial",
        name="Spatial Computing, AR & VR",
        tam_usd=230e9,
        industry_cagr=0.28,
        commercial_maturity=0.22,
        breakthrough_potential=0.75,
        catalyst_intensity=0.55,
        winner_take_most=0.80,
        plausible_winner_share=0.20,
        winner_ev_sales=9.0,
        source_note="Mid-2030s AR/VR/spatial-computing hardware plus software forecasts.",
        catalyst_archetypes=("consumer device launches and price-point breaks",
                             "enterprise deployment wins", "display/optics technology milestones"),
        keywords=("augmented reality", "virtual reality", "spatial computing",
                  "mixed reality", "waveguide display"),
        tickers=("VUZI", "KOPN", "IMMR"),
    ),
    SpeculativeTheme(
        key="digital_assets",
        name="Digital Assets & Blockchain Infrastructure",
        tam_usd=280e9,
        industry_cagr=0.25,
        commercial_maturity=0.30,
        breakthrough_potential=0.70,
        catalyst_intensity=0.85,
        winner_take_most=0.65,
        plausible_winner_share=0.18,
        winner_ev_sales=7.0,
        source_note="Mid-2030s estimates for digital-asset infrastructure revenue "
                    "(exchanges, custody, mining, tokenization) -- not total crypto "
                    "market capitalization, which is an asset value rather than a "
                    "revenue-generating market.",
        catalyst_archetypes=("regulatory clarity and legislation",
                             "institutional adoption and ETF flows",
                             "network upgrade and halving cycles",
                             "tokenization mandates from traditional finance"),
        keywords=("bitcoin", "cryptocurrency", "blockchain", "digital asset",
                  "crypto mining", "stablecoin", "tokenization"),
        tickers=("COIN", "MARA", "RIOT", "CLSK", "HUT", "BITF", "CIFR", "WULF", "MSTR", "HOOD"),
    ),
    SpeculativeTheme(
        key="ai_infrastructure",
        name="AI Compute & Infrastructure",
        tam_usd=1.4e12,
        industry_cagr=0.30,
        commercial_maturity=0.70,
        breakthrough_potential=0.85,
        catalyst_intensity=0.70,
        winner_take_most=0.85,
        plausible_winner_share=0.18,
        winner_ev_sales=10.0,
        source_note="Mid-2030s AI accelerator, data-center, networking and AI-cloud "
                    "infrastructure spend. Among the best-corroborated figures here, "
                    "given existing hyperscaler capital-expenditure disclosures.",
        catalyst_archetypes=("next-generation chip/architecture launches",
                             "hyperscaler capital-expenditure guidance",
                             "large multi-year capacity contracts",
                             "new data-center capacity coming online"),
        # NOTE: keywords must not overlap as substrings of one another
        # ("data center" / "data centers" would double-count a single
        # mention), because classify_theme scores a match by counting
        # DISTINCT keywords found.
        keywords=("artificial intelligence", "ai cloud", "ai infrastructure", "gpu",
                  "accelerated computing", "data center",
                  "high performance computing", "neural network"),
        tickers=("NVDA", "AMD", "AVGO", "MRVL", "SMCI", "VRT", "CRWV", "IREN", "NBIS", "ALAB"),
    ),
    SpeculativeTheme(
        key="ai_software",
        name="AI Software & Applications",
        tam_usd=850e9,
        industry_cagr=0.32,
        commercial_maturity=0.45,
        breakthrough_potential=0.85,
        catalyst_intensity=0.60,
        winner_take_most=0.80,
        plausible_winner_share=0.16,
        winner_ev_sales=12.0,
        source_note="Mid-2030s enterprise and consumer AI software/agent market "
                    "estimates, excluding the underlying compute infrastructure.",
        catalyst_archetypes=("large enterprise or government contract awards",
                             "model/product capability releases",
                             "seat and usage expansion within existing customers",
                             "gross-margin inflection as inference costs fall"),
        keywords=("machine learning", "generative ai", "large language model",
                  "ai platform", "ai software", "computer vision", "ai agent"),
        tickers=("PLTR", "AI", "SNOW", "PATH", "BBAI", "SOUN", "TEM"),
    ),
    SpeculativeTheme(
        key="advanced_semis",
        name="Advanced Semiconductors",
        tam_usd=1.1e12,
        industry_cagr=0.12,
        commercial_maturity=0.85,
        breakthrough_potential=0.65,
        catalyst_intensity=0.55,
        winner_take_most=0.75,
        plausible_winner_share=0.15,
        winner_ev_sales=7.0,
        source_note="Mid-2030s total semiconductor industry revenue forecasts "
                    "(commonly cited ~$1T by 2030, growing thereafter).",
        catalyst_archetypes=("process-node transitions", "design wins at major customers",
                             "fab capacity expansion", "export-control and policy changes"),
        keywords=("semiconductor", "integrated circuit", "wafer", "foundry",
                  "chip design", "photonics"),
        tickers=("TSM", "INTC", "MU", "LSCC", "AMAT", "LRCX", "KLAC", "ASML", "ARM"),
    ),
    SpeculativeTheme(
        key="biotech_therapeutics",
        name="Biotech Therapeutics",
        tam_usd=800e9,
        industry_cagr=0.10,
        commercial_maturity=0.40,
        breakthrough_potential=0.90,
        catalyst_intensity=0.95,
        winner_take_most=0.50,
        plausible_winner_share=0.08,
        winner_ev_sales=6.0,
        source_note="Mid-2030s branded-therapeutics market. Individual clinical-stage "
                    "companies address only a slice of this, which the "
                    "plausible_winner_share figure reflects.",
        catalyst_archetypes=("clinical trial data readouts", "FDA/EMA approval decisions",
                             "pharma licensing or acquisition interest",
                             "label expansions and new indications"),
        keywords=("clinical trial", "therapeutics", "biopharmaceutical", "drug candidate",
                  "phase 3", "phase 2", "oncology", "immunotherapy", "biotechnology"),
        tickers=("MRNA", "BNTX", "SRPT", "ALNY", "RXRX", "IOVA"),
    ),
    SpeculativeTheme(
        key="obesity_metabolic",
        name="Obesity & Metabolic Therapeutics",
        tam_usd=170e9,
        industry_cagr=0.22,
        commercial_maturity=0.60,
        breakthrough_potential=0.60,
        catalyst_intensity=0.90,
        winner_take_most=0.75,
        plausible_winner_share=0.25,
        winner_ev_sales=8.0,
        source_note="Early-2030s GLP-1 / obesity therapeutic market forecasts, one of "
                    "the fastest-revised categories in pharma.",
        catalyst_archetypes=("oral formulation trial results", "supply/manufacturing scale-up",
                             "payer and reimbursement coverage decisions",
                             "label expansion into adjacent indications"),
        keywords=("glp-1", "obesity treatment", "weight loss drug", "metabolic disease"),
        tickers=("LLY", "NVO", "VKTX", "ALT"),
    ),
    SpeculativeTheme(
        key="energy_storage",
        name="Batteries & Energy Storage",
        tam_usd=420e9,
        industry_cagr=0.21,
        commercial_maturity=0.50,
        breakthrough_potential=0.80,
        catalyst_intensity=0.65,
        winner_take_most=0.55,
        plausible_winner_share=0.15,
        winner_ev_sales=6.0,
        source_note="Mid-2030s battery cell plus grid-storage market forecasts.",
        catalyst_archetypes=("cell chemistry and energy-density milestones",
                             "gigafactory commissioning", "OEM supply agreements",
                             "grid-scale project awards"),
        keywords=("battery", "energy storage", "solid state battery", "lithium ion",
                  "grid storage"),
        tickers=("QS", "ENVX", "SLDP", "AMPX", "FLNC", "EOSE"),
    ),
    SpeculativeTheme(
        key="defense_tech",
        name="Defense Technology & Autonomy",
        tam_usd=420e9,
        industry_cagr=0.13,
        commercial_maturity=0.55,
        breakthrough_potential=0.70,
        catalyst_intensity=0.80,
        winner_take_most=0.60,
        plausible_winner_share=0.12,
        winner_ev_sales=7.0,
        source_note="Mid-2030s share of global defense budgets addressable by "
                    "software-defined, autonomous and unmanned systems specifically.",
        catalyst_archetypes=("program-of-record awards", "allied procurement decisions",
                             "budget appropriation cycles", "combat/field deployment validation"),
        keywords=("defense technology", "unmanned aerial", "military drone", "counter-drone",
                  "defense electronics", "hypersonic"),
        tickers=("KTOS", "AVAV", "RCAT", "ONDS", "DRS"),
    ),
    SpeculativeTheme(
        key="next_gen_comms",
        name="Next-Generation Connectivity",
        tam_usd=240e9,
        industry_cagr=0.16,
        commercial_maturity=0.45,
        breakthrough_potential=0.70,
        catalyst_intensity=0.70,
        winner_take_most=0.70,
        plausible_winner_share=0.18,
        winner_ev_sales=8.0,
        source_note="Mid-2030s direct-to-device satellite, 6G and optical-networking "
                    "market estimates.",
        catalyst_archetypes=("spectrum approvals", "carrier partnership agreements",
                             "constellation or network build-out milestones",
                             "commercial service launch"),
        keywords=("direct to device", "satellite broadband", "6g", "optical networking",
                  "broadband constellation"),
        tickers=("ASTS", "IRDM", "GSAT", "ANET", "CIEN"),
    ),
    SpeculativeTheme(
        key="ev",
        name="Electric Vehicles",
        tam_usd=1.3e12,
        industry_cagr=0.16,
        commercial_maturity=0.55,
        breakthrough_potential=0.55,
        catalyst_intensity=0.60,
        winner_take_most=0.50,
        plausible_winner_share=0.12,
        winner_ev_sales=3.0,
        source_note="Mid-2030s global EV sales value. Note the deliberately LOW "
                    "winner_ev_sales multiple: vehicle manufacturing is a "
                    "capital-intensive, low-margin business, and mature auto makers "
                    "trade at a fraction of a software company's sales multiple.",
        catalyst_archetypes=("production ramp and delivery milestones",
                             "new model launches", "gross-margin crossover to positive",
                             "manufacturing capacity commissioning"),
        keywords=("electric vehicle", "battery electric", "ev manufacturer"),
        tickers=("TSLA", "RIVN", "LCID", "NIO", "XPEV", "LI"),
    ),
    SpeculativeTheme(
        key="cybersecurity",
        name="Cybersecurity",
        tam_usd=500e9,
        industry_cagr=0.13,
        commercial_maturity=0.75,
        breakthrough_potential=0.50,
        catalyst_intensity=0.45,
        winner_take_most=0.65,
        plausible_winner_share=0.14,
        winner_ev_sales=9.0,
        source_note="Mid-2030s global information-security spend forecasts.",
        catalyst_archetypes=("platform consolidation wins", "large enterprise displacements",
                             "regulatory compliance mandates"),
        keywords=("cybersecurity", "cyber security", "endpoint security", "zero trust",
                  "threat detection"),
        tickers=("CRWD", "PANW", "ZS", "S", "OKTA", "NET"),
    ),
    SpeculativeTheme(
        key="cloud_saas",
        name="Cloud & Enterprise Software",
        tam_usd=1.5e12,
        industry_cagr=0.15,
        commercial_maturity=0.85,
        breakthrough_potential=0.45,
        catalyst_intensity=0.40,
        winner_take_most=0.70,
        plausible_winner_share=0.12,
        winner_ev_sales=8.0,
        source_note="Mid-2030s public-cloud plus enterprise-SaaS spend forecasts.",
        catalyst_archetypes=("large platform migrations", "net-revenue-retention inflections",
                             "new product-line attach"),
        keywords=("software as a service", "cloud platform", "enterprise software",
                  "saas", "cloud infrastructure"),
        tickers=("MSFT", "CRM", "NOW", "DDOG", "MDB", "ORCL"),
    ),
    SpeculativeTheme(
        key="fintech",
        name="Fintech & Digital Financial Services",
        tam_usd=600e9,
        industry_cagr=0.14,
        commercial_maturity=0.70,
        breakthrough_potential=0.50,
        catalyst_intensity=0.50,
        winner_take_most=0.60,
        plausible_winner_share=0.12,
        winner_ev_sales=6.0,
        source_note="Mid-2030s digital-payments and digital-lending revenue forecasts.",
        catalyst_archetypes=("banking-license or charter approvals",
                             "take-rate and monetization expansion",
                             "profitability inflection", "major distribution partnerships"),
        keywords=("digital payments", "fintech", "neobank", "payment processing",
                  "digital banking", "buy now pay later"),
        tickers=("SOFI", "AFRM", "UPST", "PYPL", "SQ", "XYZ", "NU", "TOST"),
    ),
)


# ---------------------------------------------------------------------------
# The fallback profile.
#
# Used when a company matches no emerging theme -- i.e. it operates in a
# mature, established industry. This is NOT a penalty applied for being
# unrecognized: the numbers describe what a mature industry genuinely looks
# like (large but slow-growing market, fully proven business model, few
# discrete repricing catalysts, low multiples). A large mature company like
# a beverage or utility business scores low on the Speculative Score for the
# correct reason -- limited potential to become several times larger -- not
# because of a data gap.
# ---------------------------------------------------------------------------
MATURE_INDUSTRY = SpeculativeTheme(
    key="mature",
    name="Mature / Established Industry",
    tam_usd=500e9,
    industry_cagr=0.04,
    commercial_maturity=0.95,
    breakthrough_potential=0.10,
    catalyst_intensity=0.20,
    winner_take_most=0.35,
    plausible_winner_share=0.12,
    winner_ev_sales=3.0,
    source_note="Generic profile for an established industry with no identified "
                "emerging-technology exposure. Represents a large but slow-growing "
                "market with proven economics and few discrete repricing events.",
    catalyst_archetypes=("cyclical demand recovery", "margin or cost-restructuring programs",
                         "capital-return policy changes"),
    keywords=(),
    tickers=(),
)


@dataclass
class ThemeMatch:
    """The outcome of classifying one company into the theme table."""

    theme: SpeculativeTheme
    #: "ticker" (explicit, high confidence), "keyword" (business-summary
    #: match, weaker), or "none" (fell back to MATURE_INDUSTRY).
    match_basis: str
    #: Other themes the company also matched, for display.
    secondary: list[SpeculativeTheme] = field(default_factory=list)

    @property
    def is_emerging(self) -> bool:
        return self.theme.key != MATURE_INDUSTRY.key


def _opportunity_rank(theme: SpeculativeTheme) -> float:
    """Rank competing theme matches by how speculative the opportunity is.

    When a company matches several themes (e.g. a company described as both
    "data center" and "bitcoin mining"), the one with the greater
    combination of market size and growth is treated as primary. Using a
    single explicit ranking function keeps the choice deterministic and
    auditable rather than depending on table ordering.
    """
    return theme.tam_usd ** 0.25 * (1.0 + theme.industry_cagr) * (1.0 + theme.breakthrough_potential)


#: Fields that describe what a company primarily IS, rather than merely
#: mentioning something it does. A match here is strong evidence.
_PRIMARY_FIELDS = ("industry", "sector", "longName", "shortName")
_PRIMARY_FIELD_WEIGHT = 3
#: Minimum weighted score required before a keyword match is accepted.
#: Set to 2 so that a SINGLE passing mention buried in a long business
#: summary is never sufficient on its own.
#:
#: This threshold exists because of a real false positive found in testing:
#: Exxon Mobil's business description lists "low-carbon data center" among a
#: dozen other business lines, which matched the AI-infrastructure theme on
#: one incidental phrase and reclassified an integrated oil major as an AI
#: company. Requiring either a primary-field match or two DISTINCT keywords
#: rejects that case without any company-specific special-casing.
_MIN_KEYWORD_SCORE = 2


def _keyword_score(theme: SpeculativeTheme, summary: str, primary: str) -> int:
    """Weighted count of distinct keyword matches for one theme.

    A keyword found in a short, definitional field (industry, sector, or
    company name) counts far more than the same word appearing once inside a
    multi-paragraph business summary, because the former states what the
    company is while the latter may only note something it happens to
    mention.
    """
    score = 0
    for kw in theme.keywords:
        if kw in primary:
            score += _PRIMARY_FIELD_WEIGHT
        elif kw in summary:
            score += 1
    return score


def classify_theme(ticker: str, info: Optional[dict] = None) -> ThemeMatch:
    """Classify a company into its most relevant speculative theme.

    Evidence tiers, strongest first:
      1. An explicit ticker listing in a theme's `tickers` tuple.
      2. A sufficiently strong keyword match (see `_MIN_KEYWORD_SCORE`)
         against the company's industry, sector, name, or business summary.
      3. No qualifying match -> MATURE_INDUSTRY.

    Note that this function only decides WHICH industry profile applies. It
    deliberately does not decide how many points the company earns from it:
    src/speculative.py gates theme-derived optionality points on independent
    financial evidence, so even a correct classification cannot manufacture
    a high score without corroborating financials. See
    `theme_evidence_strength` there.
    """
    info = info or {}
    ticker_upper = (ticker or "").strip().upper()

    ticker_hits = [t for t in THEMES if ticker_upper in t.tickers]
    if ticker_hits:
        ticker_hits.sort(key=_opportunity_rank, reverse=True)
        return ThemeMatch(theme=ticker_hits[0], match_basis="ticker", secondary=ticker_hits[1:])

    summary = str(info.get("longBusinessSummary") or "").lower()
    primary = " ".join(str(info.get(k) or "") for k in _PRIMARY_FIELDS).lower()

    scored = [(t, _keyword_score(t, summary, primary)) for t in THEMES]
    qualifying = [(t, s) for t, s in scored if s >= _MIN_KEYWORD_SCORE]
    if qualifying:
        # Sort by match strength first, then by how speculative the
        # opportunity is, so the result is deterministic and independent of
        # this table's declaration order.
        qualifying.sort(key=lambda ts: (ts[1], _opportunity_rank(ts[0])), reverse=True)
        return ThemeMatch(theme=qualifying[0][0], match_basis="keyword",
                          secondary=[t for t, _ in qualifying[1:]])

    return ThemeMatch(theme=MATURE_INDUSTRY, match_basis="none")
