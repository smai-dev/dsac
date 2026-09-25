# Causal NLP Research: Comparing Policy Intent and Stakeholder Discourse

**Project type:** Multilingual NLP, topic modelling, causal-language analysis, evidence-grounded LLM workflows  
**Research question:** How can text analysis reveal differences between AI-in-education policy language and the concerns, experiences and expectations expressed in stakeholder evidence?

## 1. Executive overview

This research developed a **multi-layered text-as-data framework** to compare AI-in-education policy documents with original stakeholder-sentiment reports. It investigates **what subjects the texts discuss, whether expressed causal claims have semantic counterparts, and how the texts connect causes, requirements, mechanisms, risks and outcomes**. An exploratory LLM-assisted stage then interprets selected evidence at sentence level.

The empirical corpus contained **56 policy documents and 17 original sentiment documents**, with Irish, French and broader European evidence. Australia and the United States also appear in the policy corpus and comparative background. Synthetic sentiment was kept **separate from observed evidence** and used for robustness tests only.

**Core achievement:** A traceable workflow combining thematic, semantic and structural causal-*language* measurements, with source-grounded interpretation. It does **not** infer real-world causal effects or measure whether a policy succeeds in practice.

## 2. Research workflow

```text
Policy PDFs + original sentiment reports       Separately marked synthetic sentiment
                  │                                        │
       layout-aware PDF → Markdown              robustness/sensitivity input only
                  ↓                                        │
       conservative cleaning + metadata                    │
                  ↓                                        │
       heading-aware passage chunking                       │
                  ↓                                        │
       LDA / BERTopic-style / NMF topic models ←────────────┘ (separate tests)
                  ↓
       freeze reference topic assignments
                  ↓
       build source-traceable, cleaned sentence inventory
             ├── semantic causal-claim coverage
             ├── causal-frame network divergence
             └── exploratory LLM analyst + verifier
                  ↓
       compare findings and inspect original sentences
```

### A. Corpus construction and provenance

**Docling-based PDF conversion** preserved headings, reading order and document structure in Markdown. Conservative cleaning removed extraction artefacts without rewriting substantive content. Document and passage records retained source identifiers, corpus role, country and text location. Reports were segmented into **heading-aware passages** (target **350 words**, overlap **50 words**) so one long document would not collapse several topics into a single representation.

A common causal-analysis inventory then cleaned and deduplicated sentences *within each document* while retaining links to their passages and source documents. The reported inventory contained **15,281 policy sentences** and **4,182 original-sentiment sentences**.

**Learning outcome:** Build NLP datasets that preserve a route from every model output back to the underlying source, and recognise how chunking, overlapping passages and deduplication influence downstream measurements.

### B. Topic modelling: where do the corpora overlap?

Three complementary approaches were compared:

| Method | Practical role |
| --- | --- |
| **LDA** | A probabilistic lexical baseline: each document is a mixture of word-distribution topics. |
| **BERTopic-style semantic clustering** | Groups passage embeddings and assigns interpretable topic terms. |
| **Non-negative Matrix Factorisation (NMF)** | Factorises a non-negative text matrix into passage–topic and topic–term representations suitable for consistent downstream projection. |

The retained global **policy** topic counts were **4 (LDA), 8 (BERTopic-style), and 9 (NMF)**; the **original-sentiment** counts were **3, 9 and 9**, respectively. These were selections within the study's model comparisons, not universal optimum topic counts. **NMF** provided the frozen reference topic spaces for subsequent causal analyses because of its explicit topic weights and ability to project later evidence into a fixed representation; it was **not** declared superior on every metric.

**Learning outcome:** Contrast lexical, embedding-based and matrix-factorisation approaches; inspect topic coherence, diversity, balance and interpretability; keep policy and original-sentiment topic identities distinct until a comparison explicitly projects them into a shared space.

## 3. Two independent ways to compare causal language

Here **“causal” describes relations *asserted in text***—such as “training enables responsible AI use”—rather than an experimentally identified cause of an outcome. Both methods work from the same cleaned sentence inventory but answer different questions.

### Method 1 — Semantic causal-claim coverage

English/French cue rules identify eligible sentences containing causal or conditional language and usable cause/effect spans. Each eligible **full sentence** becomes a claim. A multilingual sentence-embedding model represents claims as vectors, then compares each policy claim with its **three nearest original-sentiment claims**, and vice versa.

A claim's coverage is the average cosine similarity to those neighbours; its **deficit = 1 − coverage**. Equal-size, repeated sampling helps control the substantially different sizes of the policy and sentiment claim pools. Direction here means **which corpus supplies the query claims**; it does **not** mean policy causes sentiment.

**Reported evidence:** **3,930 policy** and **761 original-sentiment claims** were extracted. Across **50 balanced resamples**, mean policy→sentiment similarity was **0.6415** (deficit **0.3585**); sentiment→policy similarity was **0.6325** (deficit **0.3675**). The study also inspected differences by retained topic and, where sufficient evidence existed, by country. Its low-similarity cutoff was an **inspection threshold, not a validated match/no-match classifier**.

**Learning outcome:** Construct directional nearest-neighbour comparisons, account for uneven corpora, use multilingual embeddings, and distinguish a numerical semantic match from an explicitly verified equivalence of causal claims.

### Method 2 — Causal-frame network divergence

A different pipeline extracts structured frames of the form:

```text
(cause or prerequisite) --[normalised relation family]--> (effect or outcome)
```

The six relation families cover **causes/increases; reduces/prevents; enables/supports; requires/depends on; risks/threatens; expected improvement**. After frame-quality checks and within-document relation deduplication, cause and effect spans are separately projected into the **frozen policy NMF topic space**. Weighted, directed topic-to-topic networks are then compared for policy and original sentiment using **Jensen–Shannon divergence** and topic/relation-family summaries.

**Reported evidence:** The primary comparison used **3,196 policy** and **655 sentiment frames** and reported network divergence **0.2596**. Alternative source-balancing, frame-quality and topic-projection specifications were examined, along with source-level resampling and a separate synthetic-data sensitivity test. Policy text gave relatively more weight to enabling mechanisms, requirements and intended improvements; original sentiment gave relatively more weight to expressed causes and consequences. Those are *distributional differences in extracted statements*, not stronger or weaker real-world causal effects.

**Learning outcome:** Transform unstructured sentences into directional, typed relations; normalise relation vocabularies; project spans into a shared fixed taxonomy; compare weighted networks; and test sensitivity to extraction and projection choices.

## 4. Evidence-grounded LLM analysis

An **exploratory** third stage sampled original source sentences directly—not the outputs of the preceding causal methods—and used distinct **analyst** and **verifier** prompts. It repeated analyses over source-balanced batches, matched recurring findings across runs and retained findings only after evidence, recurrence, agreement, confidence, faithfulness and stability filters. Source-sentence identifiers made each accepted interpretation auditable.

**Reported result:** **16 recurrent findings** survived filtering: **13 partial alignments**, **2 sentiment gaps** and **1 alignment**. Of **31 evidence sentences cited** by the retained findings, **18** were outside the fixed cue-based claim inventory. This illustrates complementary access to less formulaic relationships; it does **not** establish superior LLM accuracy or recall. The LLM stage covered **sampled evidence**, and it lacked a completed human-labelled precision/recall evaluation.

**Learning outcome:** Design distinct analyst/verifier stages, impose source-grounding and recurrent-evidence filters, quantify stability across runs, and state where human validation remains necessary.

## 5. Synthetic data as a robustness check—not evidence

Synthetic sentiment addressed a **testing** need arising from uneven source availability. Lexical, semantic, distributional and classifier-based checks evaluated how closely it resembled original sentiment; the study found that generated text remained distinguishable from original reports. Synthetic records therefore retained separate labels and were **never used to claim what stakeholders actually believed**.

**Learning outcome:** Separate data augmentation or perturbation for robustness from the evidence used for empirical conclusions. Preserve provenance and check whether results change when the input distribution changes.

## 6. Principal findings and research boundaries

The combined analyses found **thematic correspondence alongside selective differences** in causal-claim coverage and causal-frame organisation. Differences were particularly visible around **learning and assessment, curriculum development and school oversight**. Related themes could appear in both corpora yet occupy different causal roles. The LLM stage supplied sentence-level examples of relationships missed by fixed cues and questionable superficial cue matches.

These outputs examine **an observable policy–sentiment discourse gap**, not classroom implementation itself. Country-level sentiment evidence was sparse and uneven, fixed-rule extraction and topic projection introduce modelling assumptions, and the LLM evaluation was exploratory. No real-world causal effect, policy effectiveness, representative national result or across-the-board LLM superiority was established.

## 7. Skills and learning outcomes for a technical portfolio

**Data engineering and reproducibility:** layout-aware PDF extraction, Markdown/CSV pipelines, heading-aware chunking, source metadata, versioned sentence inventories and deduplication.  
**Multilingual NLP:** text representations, TF-IDF, sentence embeddings, semantic similarity, English/French cue-based extraction and topic modelling (LDA, BERTopic-style, NMF).  
**Structural analysis:** cause–relation–effect frames, directed weighted graphs, shared-topic projection, Jensen–Shannon divergence and cross-method interpretation.  
**Experimental design:** source balancing, repeated sampling, sensitivity analysis, synthetic-data validation, ablations/specification checks and honest handling of missing ground truth.  
**LLM workflow design:** evidence-based prompting, separate analysis/verification, cross-run recurrence, faithfulness checks and explicit human-review boundaries.

**Recruiter-facing summary:** Research prototype integrating multilingual corpus processing, topic models, semantic causal-claim matching, directed causal-frame networks and evidence-grounded LLM verification to analyse differences between policy language and stakeholder discourse while preserving source traceability and clearly separating expressed causal claims from real-world causal inference.
