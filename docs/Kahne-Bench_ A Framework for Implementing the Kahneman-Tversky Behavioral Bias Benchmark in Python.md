# **Kahne-Bench: A Framework for Implementing the Kahneman-Tversky Behavioral Bias Benchmark in Python**

### **1\. Introduction: The Mandate for a New Generation of Bias Benchmarking**

As Large Language Models (LLMs) transition from experimental tools to integrated components of high-stakes professional workflows—from managerial decision-making to healthcare—the imperative to understand and mitigate their cognitive vulnerabilities has never been more critical. The rationality of their judgments is no longer an academic curiosity but a prerequisite for trustworthy and responsible deployment. To meet this challenge, we require evaluation frameworks that move beyond surface-level performance metrics and probe the deep-seated cognitive patterns that govern model behavior. A robust methodology must be grounded in a coherent theory of cognition, capable of identifying the systematic deviations from rational thought that have been extensively documented in humans.

This document introduces Kahne-Bench, a novel benchmarking framework uniquely grounded in the foundational "two-system" view of cognition articulated by Nobel laureate Daniel Kahneman. This dual-process theory distinguishes between the fast, automatic, and intuitive operations of **System 1** and the slow, serial, and deliberately controlled operations of **System 2**. Human judgment often relies on the heuristics of System 1, which, while efficient, can lead to predictable errors or "cognitive biases." Kahne-Bench is meticulously designed to create scenarios that elicit these two modes of thought within LLMs, measuring their susceptibility to the same cognitive illusions that affect human reasoning and their capacity for deliberate, corrective thought.

Existing benchmarks have paved the way, but they reveal critical limitations. Early work like `CogBench` adapted methodologies from psychology labs but focused on general behavioral phenotyping rather than the specific, theoretically unified biases of the Kahneman-Tversky research program. Subsequent efforts, while more targeted, have been constrained in scope; `CoBBLEr` evaluated only six biases and relied on a potentially biased LLM-as-evaluator methodology, while the work of `Saeedi et al.` was limited to nine biases across three models. The most comprehensive existing dataset, developed by `Malberg et al.`, provides a valuable framework for testing 30 distinct biases. Kahne-Bench represents the necessary evolution of this work. Our ambitious goal is to expand this scope to cover 50 distinct biases, justifying this expansion by systematically operationalizing the full breadth of Kahneman and Tversky's research, including areas like prototype heuristics and extension neglect that remain under-explored in current LLM benchmarks. Crucially, Kahne-Bench unifies all test cases under a single theoretical umbrella with a direct lineage to Prospect Theory and dual-process theory.

The purpose of this document is to provide a comprehensive technical framework for the design and implementation of Kahne-Bench. We will first detail its core architectural pillars, which ensure unprecedented scope, methodological rigor, and ecological validity. We will then transition to a practical guide for its implementation in Python, outlining the core data structures, generative engines, and evaluation pipelines. Finally, we will explore the benchmark's advanced features and its planned evolution into a dynamic training environment for actively cultivating rationality in next-generation AI systems.

### **2\. The Kahne-Bench Architectural Blueprint**

A benchmark's value is determined by the coherence and rigor of its underlying architecture. Kahne-Bench is built upon five core design pillars that differentiate it from its predecessors, ensuring comprehensiveness, methodological innovation, and real-world relevance. These pillars are engineered not only to detect the presence of bias but to quantify its magnitude, consistency, and context-sensitivity, providing a multi-dimensional "cognitive fingerprint" for any LLM.

#### **Pillar 1: Unprecedented Scope and Theoretical Purity**

Kahne-Bench aims to provide the most exhaustive evaluation of cognitive biases to date, encompassing **50 distinct biases** drawn directly from the research program of Kahneman and Tversky. Unlike benchmarks that measure a broad but disconnected set of behavioral phenomena, every test case in Kahne-Bench is theoretically anchored. By unifying the entire framework under the principles of **prospect theory** and **dual-process theory**, we move beyond the descriptive cataloging of predecessors. This theoretical purity allows for a mechanistic understanding of *why* errors occur—for instance, by identifying attribute substitution as the root cause of a heuristic judgment, which is a fundamentally deeper scientific objective. Recent empirical research has identified several biases that demonstrate particularly distinctive behavior in LLMs and must be explicitly included in our taxonomy: Primacy Bias, where models preferentially select options presented early in prompts (options A/B over C/D); Status Quo Bias, which interestingly manifests inversely in LLMs compared to humans; Group Attribution Bias, affecting assessments based on demographic attributes like gender; the Halo Effect, leading to overgeneralization of positive traits; and Neglect of Probability, particularly pronounced in emotionally-charged scenarios where models struggle to integrate statistical reasoning with vivid contextual information.

#### **Pillar 2: Multi-Domain Ecological Validity**

To be truly meaningful, bias evaluation must extend beyond abstract puzzles to reflect the complex decision-making contexts where LLMs are deployed. Kahne-Bench ensures ecological validity by generating test cases across five critical, real-world domains:

* **Individual Decisions:** Personal finance, consumer choice, and lifestyle planning.  
* **Professional Judgments:** Managerial strategy, medical diagnosis, and legal analysis.  
* **Social Interactions:** Negotiation, persuasion, and collaborative problem-solving.  
* **Temporal Decisions:** Long-term planning, investment horizons, and delayed gratification.  
* **Risk Assessment:** Evaluating uncertainty in policy, technology, and environmental domains.

For example, the **anchoring** bias is tested not just with abstract numerical estimation but in a salary negotiation scenario. **Loss aversion** is evaluated through investment portfolio decisions, while the **framing effect** is measured by an LLM's response to public policy proposals phrased in terms of gains versus losses. This multi-domain mapping ensures the benchmark measures how biases manifest in practical, high-stakes situations.

#### **Pillar 3: Multi-Scale Methodological Innovation**

Kahne-Bench introduces a novel multi-scale testing methodology to provide a holistic view of an LLM's cognitive profile, moving from isolated tests to complex, interactive assessments.

| Scale | Description |
| :---- | :---- |
| **Micro** | Measures the presence and magnitude of a single, isolated bias using a control vs. treatment paradigm (e.g., assessing the Framing Effect with a single decision). |
| **Meso** | Evaluates bias interactions and compounding effects by triggering multiple biases within a single, complex scenario (e.g., Anchoring \+ Availability). |
| **Macro** | Tests the persistence of a bias across a sequential chain of related decisions to measure its stability and the LLM's learning capacity over time. |
| **Meta** | Assesses the LLM’s capacity for self-correction and debiasing by measuring its response to explicit instructions to engage in System 2 reasoning or identify its own cognitive errors. |

#### **Pillar 4: Dynamic and Quantified Bias Measurement**

Existing benchmarks have largely focused on a static, binary classification of bias—an approach that fails to capture how biases evolve or persist over a sequence of related decisions. Kahne-Bench directly addresses this limitation by enabling dynamic and quantified measurement. Through sequential decision chains (the Macro scale), the framework can track how a bias evolves as new information is introduced. Furthermore, it quantifies the *strength* of a bias by systematically varying the intensity of the bias trigger. This allows for a more nuanced understanding, distinguishing between a mild cognitive inclination and a deeply entrenched, irrational pattern of thought that resists correction.

#### **Pillar 5: Advanced, Multi-faceted Evaluation Metrics**

Moving beyond simple accuracy, Kahne-Bench will employ a suite of six advanced metrics designed to capture a comprehensive picture of an LLM's decision-making profile:

* **Bias Magnitude Score (BMS):** Quantifies the strength of a given bias by measuring the degree of deviation between the model's response in a treatment condition and the rational baseline established in the control condition.  
* **Bias Consistency Index (BCI):** Measures how consistently a model exhibits a particular bias across different domains and contexts, indicating whether the bias is a sporadic error or a systematic flaw.  
* **Bias Mitigation Potential (BMP):** Assesses the model's ability to overcome a demonstrated bias when provided with explicit debiasing prompts or chain-of-thought instructions, measuring its capacity for System 2 override.  
* **Human Alignment Score (HAS):** Compares the LLM's pattern of biases to established patterns in specific human cohorts. This is motivated by meta-analyses showing that human bias patterns are not monolithic but vary with factors like age. HAS thus determines whether the model's irrationality mirrors specific human cognitive shortcuts or represents a uniquely artificial form of error.  
* **Response Consistency Index (RCI):** Measures the variance in model responses across multiple identical trials of the same test case. Research by Saeedi et al. (2024) demonstrates that LLMs show significant response inconsistency, with some models contradicting themselves frequently. A model showing 50% bias susceptibility could actually be highly inconsistent rather than consistently biased—this distinction is methodologically critical for distinguishing systematic cognitive patterns from stochastic noise.  
* **Calibration Awareness Score (CAS):** Inspired by the calibration metrics used in Humanity’s Last Exam benchmark, this metric measures whether a model recognizes when it is being influenced by a cognitive bias. CAS compares the model’s stated confidence in its reasoning against its actual susceptibility to bias triggers. A model that is 50% biased but 90% confident it is unbiased represents a greater risk than one that acknowledges uncertainty, making metacognitive awareness a crucial safety-relevant metric.

Together, these five pillars establish an architectural vision for a benchmark that is not only more comprehensive but also methodologically deeper, leading to the practical implementation steps that follow.

### **3\. Core Implementation in Python: The `kahne_bench` Module**

This section provides a practical, hands-on guide to building the Kahne-Bench framework in Python. The architecture is designed to be modular and scalable, enabling researchers to easily generate vast datasets, execute tests on new models, and compute our advanced evaluation metrics. The functional structure, particularly the separation of generative and evaluative engines, draws inspiration from successful implementation patterns seen in projects like `simonmalberg/cognitive-biases-in-llms`.

#### **3.1. Project Setup and Dependencies**

To begin developing with Kahne-Bench, a developer would follow these initial setup steps:

* **Create a Python Virtual Environment:** Isolate project dependencies to ensure a clean and reproducible workspace.  
* **Install Core Dependencies:** The framework relies on several key libraries for interacting with LLM APIs, data manipulation, and executing tests.  
* This core set allows for interaction with models from major providers and leverages `pandas` for efficient data handling and analysis of results.

#### **3.2. Defining Core Data Structures**

At the heart of the benchmark is a robust data structure capable of representing the complexity of our multi-scale testing methodology. The primary Python class, `CognitiveBiasInstance`, encapsulates all elements of a single test.

This class structure is designed to be highly expressive, allowing for the systematic creation of nuanced test cases. Each attribute serves a specific function that directly maps to the architectural pillars:

* `base_scenario`: Provides the neutral, unbiased context for the decision.  
* `bias_trigger`: The specific linguistic or contextual manipulation designed to elicit a cognitive bias (e.g., a high anchor value, a loss-framed outcome).  
* `control_condition`: The baseline version of the test case, designed to elicit a rational or unbiased response.  
* `treatment_conditions`: A list of variants that apply the `bias_trigger` with increasing intensity (e.g., `["weak", "moderate", "strong", "adversarial"])` to enable the calculation of the **Bias Magnitude Score (BMS)**. The “adversarial” intensity level is specifically designed to test robustness under compound bias pressure, combining multiple bias triggers from the Bias Interaction Matrix with varying intensities to simulate challenging real-world scenarios where cognitive biases interact synergistically.  
* `cross_domain_variants`: A list of scenarios adapted to different professional or social fields (e.g., `["finance", "health", "social"]`) to provide the data for the **Bias Consistency Index (BCI)**.  
* `debiasing_prompts`: A list of prompts designed to engage System 2 reasoning, used to calculate the **Bias Mitigation Potential (BMP)**.

#### **3.3. The Generative Engine: Scaling Test Case Creation**

To generate a large and diverse dataset of test cases, Kahne-Bench employs a generative engine built on the four-entity framework described by Malberg et al.: `Test Case`, `Scenario`, `Decision Result`, and `Template`. A powerful generator LLM, such as `GPT-4o`, is used to programmatically populate template gaps. This process is driven by a standardized prompt (the `GEN` prompt), which instructs the LLM to fill gaps based on a given high-level `Scenario` (e.g., "a cardiologist deciding on a treatment plan"). This method ensures the creation of thousands of unique yet structurally consistent test cases. The diversity of the generated dataset is rigorously validated using metrics like **Self-BLEU** and **ROUGE** scores to ensure low n-gram overlap and high semantic variety.

#### **3.4. The Evaluation Engine: Executing Tests and Computing Metrics**

The evaluation engine operates in a two-step process to elicit and score a target model's decision. First, the `DEC` prompt is used to present a generated test case instance to the target LLM and request a decision. The LLM is free to reason before providing its final answer. Second, to enable automated scoring, a subsequent, simpler prompt is used to extract only the model's final choice (e.g., "Option A") from its full-text response.

The extracted answers are then used to compute the advanced metrics defined in Section 2.5. For the **Bias Magnitude Score (BMS)**, Kahne-Bench builds upon foundational quantification methods. For example, Malberg et al. use a simplified metric to capture the relative shift in a model's answer: `m(a1,2,k) = k * (a1 - a2) / max(a1, a2)` where `a1` and `a2` are the answers in the control and treatment conditions, respectively. Kahne-Bench's BMS will advance this baseline by incorporating more sophisticated normalizations, such as accounting for the intensity of the bias trigger or integrating model-reported confidence scores. This allows us to move beyond simple right/wrong scoring and capture the *degree* to which a model's judgment is distorted by a cognitive bias.

### **4\. Advanced Features and Differentiators**

This section details the unique research contributions and technical features of Kahne-Bench that are largely absent in prior work. These differentiators focus on moving beyond the analysis of isolated biases to explore the more complex, dynamic, and robust nature of cognitive patterns in LLMs.

#### **4.1. The Bias Interaction Matrix**

A significant limitation of current benchmarks is that they test for biases in isolation. However, in real-world decision-making, multiple cognitive shortcuts often interact, leading to compound effects, amplifications, or even feedback loops. Kahne-Bench is the first benchmark to systematically test for these interactions. The framework implements a "Bias Interaction Matrix" that combines triggers for different biases within a single, coherent scenario. This allows for the investigation of complex cognitive phenomena, such as:

* **Anchoring x Availability:** How does the ease of recalling information (Availability Heuristic) influence an LLM's susceptibility to an initial numerical anchor?  
* **Framing x Loss Aversion:** Does a loss-framed scenario amplify the inherent bias of loss aversion, leading to extremely risk-averse behavior?  
* **Confirmation x Overconfidence:** Does a model's tendency to seek confirming evidence lead to an inflated and unjustified sense of confidence in its conclusions?

#### **4.2. Temporal Dynamics and Context Sensitivity**

Biases are not static; they can emerge, strengthen, or fade over time and in response to changing contexts. Kahne-Bench incorporates a methodology for testing these temporal and contextual dynamics by evaluating model responses under different conditions:

* **`Immediate`:** The model must provide an intuitive, "System 1" response with minimal deliberation time.  
* **`Deliberative`:** The model is given an opportunity for extended reasoning, simulating "System 2" thought.  
* **`Persistent`:** The bias is tested across a sequence of related prompts to see if it endures.  
* **`Adaptive`:** After exhibiting a bias, the model is given corrective feedback, and its ability to adapt its subsequent reasoning is measured.

Furthermore, the framework includes tests for context sensitivity, examining how factors like domain expertise (prompting the model to act as a novice vs. an expert), formality of the setting, and the perceived stakes of the decision influence the expression of bias.

#### **4.3. Adversarial Robustness and Debiasing Potential**

A core goal of Kahne-Bench is to assess the stability of identified biases and the model's potential for debiasing. To this end, the framework integrates three key robustness tests. We recognize that simple adversarial methods like paraphrasing can be insufficient and may alter the cognitive task itself. Therefore, our tests are designed to preserve the *psychological fidelity* of the bias trigger while varying linguistic form.

* **Resistance to prompt variations:** Test cases are subjected to semantic-preserving paraphrasing to ensure the detected bias is not merely an artifact of specific wording.  
* **Response to explicit debiasing instructions:** The model is explicitly warned about a potential bias (e.g., "Be careful not to be influenced by the initial number provided") to measure its ability to consciously override a heuristic.  
* **The effect of chain-of-thought reasoning:** The framework compares bias  
* **Self-help debiasing potential:** Inspired by the BiasBuster framework’s findings that high-capacity models can effectively debias themselves, this test evaluates the model’s ability to rewrite its own prompts to remove bias triggers. The model is presented with a biased prompt and asked to identify and neutralize the cognitive manipulation before responding, measuring its metacognitive debiasing capabilities. manifestation in direct-answer prompts versus prompts that explicitly require a step-by-step reasoning process, testing whether structured thinking mitigates cognitive errors.

These advanced features collectively push the boundaries of LLM evaluation. The detailed cognitive "fingerprints" they generate—capturing interaction effects and temporal dynamics—reveal complex, stateful behaviors that a static benchmark cannot fully address, thus mandating the shift to a dynamic, interactive training environment.

### **5\. Phase II: The Rationality Alignment Environment (RAE)**

The ambition of Phase II is to evolve the static Kahne-Bench evaluation suite into a dynamic, interactive Reinforcement Learning (RL) environment. This **Rationality Alignment Environment (RAE)** is designed to go beyond merely diagnosing biases and instead actively train an LLM policy that aligns with prescriptive rationality. The ultimate goal is to fine-tune an agent that learns to systematically override its flawed, human-like intuitions in favor of verifiable logical and statistical reasoning.

#### **5.1. Formalizing Scenarios as an Interactive Text Game**

The immense, open-ended action space of an LLM (i.e., any possible text generation) makes traditional RL approaches computationally infeasible. To overcome this, the K\&T scenarios within the RAE are formalized as an interactive "text game." In this paradigm, natural language describes the environment, and the LLM agent must navigate the problem by selecting from a constrained set of actions to reach a solution. The key components are:

* **State Space (S):** The state is represented by a natural language description of the current problem context, including all information presented to the agent so far.  
* **Action Space (A):** The action space is a discrete, finite set of cognitive or textual decision steps that guide the reasoning process. This transforms the problem from pure generation to structured decision-making. Examples of actions include: `{"Query for missing base rate", "Apply Bayesian calculation", "Calculate expected utility", "Propose option A"}`.

For instance, a single turn in the "Asian Disease Problem" might look like this: `State: "You are presented with a scenario where 600 people are at risk from a disease and must choose between two programs..."` `Action: "Calculate expected utility."` `New State: "...Expected utility for Program A is 200 lives saved. Expected utility for Program B is (1/3 * 600) = 200 lives saved. The programs are objectively equivalent."`

#### **5.2. Architecting an Objective Reward Model (RM)**

The success of the RAE hinges on its reward model (RM), which must be fundamentally different from those used in typical Reinforcement Learning from Human Feedback (RLHF). RLHF RMs are trained to predict human preferences, which are themselves susceptible to the very cognitive biases we aim to eliminate. Therefore, the RAE's reward model must be trained exclusively on the **mathematically derived objective ground truth** from Kahneman and Tversky's work (e.g., maximizing expected utility, adhering to Bayesian probability).

To prevent "reward hacking," the RAE will incorporate a **Counterfactual Reward Model (CRM)**. This model performs a causal intervention, computationally answering the question: “What would the reward have been if the spurious feature (e.g., gain vs. loss framing) were different?” By neutralizing the causal influence of superficial linguistic cues—like the difference between gain-framing ("200 people will be saved") and loss-framing ("400 people will die")—the CRM ensures the agent is rewarded for logically sound reasoning, not for sensitivity to manipulative language. Importantly, the CRM can be extended to incorporate model-generated counterfactuals, leveraging the “self-help” debiasing approach demonstrated by BiasBuster, where models autonomously rewrite prompts to mitigate cognitive bias. This allows the system to generate a broader and more diverse set of counterfactual scenarios beyond researcher-specified ones, potentially discovering bias triggers that human designers might overlook.

#### **5.3. Policy Optimization for Verifiable Rationality**

In the final stage, a reinforcement learning algorithm, such as Proximal Policy Optimization (PPO) or Direct Preference Optimization (DPO), is used to fine-tune the LLM agent. The agent's policy is optimized to maximize the cumulative reward issued by the objective RM. By interacting with the text-game environment over thousands of episodes, the agent learns a "policy of rationality." This policy demonstrably favors reasoning paths that are statistically and logically sound, effectively training the model to engage its "System 2" capabilities and correct the flawed initial intuitions generated by its "System 1."

The RAE thus provides a complete, end-to-end mechanism to not only measure but to actively correct cognitive biases, marking a critical step toward building more robust and trustworthy LLM agents.

### **5.4. Limitations and Methodological Considerations**

While Kahne-Bench represents a significant advancement in bias evaluation methodology, several important limitations warrant acknowledgment. First, the benchmark fundamentally measures explicit, observable outputs and cannot capture implicit or unconscious biases that may influence model behavior without manifesting in generated text. Second, the benchmark methodology itself introduces potential confounds: prompt phrasing, even when carefully designed, may inadvertently trigger or suppress biases in ways that do not reflect the model’s underlying cognitive patterns. Researchers should exercise caution when interpreting results, recognizing that bias detection is inherently sensitive to experimental framing.

Third, a critical interpretive challenge arises from the fact that some responses flagged as “biased” may actually represent contextually appropriate reasoning. For instance, a model exhibiting apparent anchoring behavior might be appropriately weighting relevant prior information rather than displaying irrational bias. Distinguishing between adaptive heuristics and maladaptive biases requires careful consideration of the decision context and cannot be fully automated.

Finally, meaningful interpretation of AI bias requires comparison to human baselines. Without understanding how human experts perform on identical tasks, we cannot determine whether a model’s bias profile is problematic or simply mirrors human cognitive patterns. Future work should partner with psychology researchers to establish rigorous human baseline data for each test case, enabling a more nuanced understanding of whether AI systems are becoming more or less rational than their human counterparts.

### 

### **6\. Conclusion and Future Directions**

The Kahne-Bench framework represents a significant advancement in the evaluation and alignment of Large Language Models. By grounding our methodology in the Nobel Prize-winning work of Kahneman and Tversky, we have created a benchmark with unparalleled theoretical purity and scope. Its key differentiators—including multi-domain and multi-scale testing, quantified bias measurement, and the systematic analysis of bias interactions—provide a far deeper and more nuanced understanding of LLM cognition than was previously possible. Furthermore, its extension into the Rationality Alignment Environment (RAE) charts a clear path from static evaluation to dynamic, RL-based training, offering a novel mechanism for engineering verifiably rational AI agents.

This research program is poised to make four key theoretical contributions to the field of AI alignment and safety:

1. **Bias Taxonomy for AI:** It will allow for the first systematic classification of which deeply ingrained human cognitive biases demonstrably transfer to LLMs and which do not, creating a foundational taxonomy for AI psychology (e.g., systematically documenting that LLMs are highly susceptible to the Framing Effect but demonstrate near-perfect immunity to the Gambler's Fallacy).  
2. **Mechanistic Understanding:** By correlating the presence and magnitude of specific biases with model architecture, scale, or training data, this work can offer insights into the underlying mechanisms that give rise to these cognitive patterns.  
3. **Systematic Debiasing Strategies:** The framework provides an empirical testbed for validating the effectiveness of various mitigation techniques, from simple prompting strategies to complex reinforcement learning policies.  
4. **Model-Specific Bias Profiles:** Kahne-Bench will enable the creation of unique "cognitive fingerprints" for different LLMs, highlighting their specific cognitive strengths and weaknesses to guide their responsible deployment in sensitive domains (e.g., demonstrating that Model X is prone to anchoring on numerical data while Model Y is more susceptible to base-rate neglect in narrative scenarios)  
   Following the successful tiered approach pioneered by SWE-bench (with its Verified, Lite, and Full variants), we propose structuring Kahne-Bench into three complementary evaluation tiers: Kahne-Bench Core provides a focused baseline of 50 biases across 5 domains for efficient initial assessment; Kahne-Bench Extended expands to the full 50 biases across all 5 domains, ensuring comprehensive ecological validity; and Kahne-Bench Interaction specifically targets bias pairs from the Bias Interaction Matrix to evaluate compound cognitive effects. Additionally, to address potential data contamination—since LLMs may have encountered classic bias scenarios like the Linda problem or Asian Disease Problem during training—the benchmark incorporates procedurally generated novel scenarios that preserve the psychological structure of bias triggers while presenting them in previously unseen contexts..

Ultimately, the development of artificial general intelligence demands a rigorous science of the artificial mind. The work outlined in this document is a crucial step in that direction, providing the tools not only to understand the cognitive illusions of today's LLMs but to build the truly trustworthy and rational AI systems of tomorrow.

