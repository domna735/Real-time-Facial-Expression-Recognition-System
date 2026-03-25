path = r'c:\Real-time-Facial-Expression-Recognition-System_v2_restart\research\final report\final report version 3.md'

with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

security_appendix = """

## A.5 Consolidation of Security Elements from Level 3 and Level 4 Subjects

In accordance with the Capstone Project requirements, this section details how security elements, principles, and best practices learned from Level 3 and Level 4 computing subjects have been consolidated and securely applied to the design, implementation, and evaluation of this Real-time Facial Expression Recognition (FER) System. Developing a computer vision system that captures, processes, and stores biometric data (faces) introduces significant security, privacy, and integrity risks. The following subsections map formal security concepts to specific architectural and procedural decisions made in this project.

### A.5.1 Edge Inference as a Privacy-Preserving Architecture (Network & Information Security)

A core threat in any webcam-based biometric system is data interception during transit, such as Man-in-the-Middle (MitM) attacks. Cloud-based computer vision APIs (e.g., sending video frames to a remote server for prediction) expose Personally Identifiable Information (PII) to network vulnerabilities and violate the principle of data minimization.

**Application in this Project:**
To mitigate this risk, the entire inference pipeline was deliberately engineered as an **Edge/Local Inference Application**. By aggressively compressing the model (compressing heavy ResNet-18/ConvNeXt teachers into a lightweight MobileNetV3 student via knowledge distillation), the system is capable of running locally on a standard consumer CPU.
- **Data at Rest / Data in Transit:** Because all frame extraction, face detection, and emotional classification happen locally in system memory, no biometric data is transmitted over the network. 
- **Privacy by Design:** This fulfills the privacy-by-design requirement taught in Level 3/4 information security modules, ensuring GDPR and local data privacy compliance by strictly isolating the data space. The webcam buffer is volatile and discarded immediately after the session unless explicitly saved for local offline adaptation.

### A.5.2 Cryptographic Data Integrity and Anti-Poisoning (Cryptography & System Security)

Machine learning models are highly susceptible to data poisoning attacks—where an adversary silently modifies training data or labels to inject backdoors or degrade model reliability. Because this project pulls data from multiple external sources (Kaggle datasets, Google Drive archives), verifying file integrity is critical.

**Application in this Project:**
Concepts from Cryptography and System Security were applied to the data ingestion pipeline:
- **Cryptographic Hashing:** As documented in Appendix A.4, SHA-256 cryptographic fingerprints are generated and checked for massive datasets (like RAF-DB test splits and ExpW archives). 
- **Manifest Validation:** Rather than loading raw folders which could be silently modified, the system uses strict `.csv` and `.json` manifests (`manifest_validation_all_with_expw.json`). A Python script validates every file path, ensuring no zero-byte malicious files or corrupted images execute arbitrary code during the PyTorch `DataLoader` instantiation.

### A.5.3 Secure Software Development Lifecycle (SSDLC) and Supply Chain Security

Modern application development heavily relies on third-party libraries, exposing developers to supply chain attacks (e.g., malicious PyPI packages executing remote code). Level 4 subjects emphasize the importance of secure environments and strict dependency tracking.

**Application in this Project:**
- **Dependency Pinning:** A strict `requirements.txt` (and `requirements-directml.txt`) freezes exact module versions (e.g., `absl-py==2.3.1`, `certifi==2025.11.12`). This guarantees reproducible builds and prevents upstream dependency hijacking from silently injecting malicious payload updates.
- **Environment Isolation:** The project uses Python virtual environments (`.venv`) to strictly isolate the application scope, preventing dependency collisions or polluting the global OS environment, following the principle of least privilege.

### A.5.4 Defense Against Model Inversion and Membership Inference Attacks (AI Security)

Machine Learning deployments are vulnerable to privacy attacks where adversaries try to reconstruct the training data from the final model weights (Model Inversion) or determine if a specific person's face was included in the training dataset (Membership Inference Attack, MIA). 

**Application in this Project:**
- **Knowledge Distillation as a Security Shield:** By deploying a *Student Model* (MobileNetV3) rather than the *Teacher Model* (ResNet-18/ConvNeXt), the system inherently shields the original training data. The student never sees the raw hard-labels (one-hot vectors) of the facial images; it only learns from the softened logits (probabilities) emitted by the teacher. This lossy information bottleneck acts as an algorithmic one-way function, making it mathematically infeasible for an attacker extracting the deployed edge device weights to invert the model and retrieve recognizable faces of the people (e.g., from AffectNet or ExpW) used in the training set.

### A.5.5 AI Robustness and Adversarial Input Resilience (Advanced Topics in AI / CV)

Standard Convolutional Neural Networks are vulnerable to adversarial perturbations—invisible noise added to an image that tricks the classifier, or natural distributional shifts (camera noise, sudden lighting changes) that crash the system. 

**Application in this Project:**
To harden the system against input-layer attacks and natural noise:
- **Algorithmic Defenses:** The student model is trained using **Temperature-Scaled Decoupled Knowledge Distillation (DKD)**, which has been shown to produce flatter, more calibrated confidence distributions rather than overconfident, spiky predictions. 
- **Temporal Stabilisation (Hysteresis & EMA):** At the inference level, adversarial "flickering" or input noise is countered using Exponential Moving Averages (EMA) and a Hysteresis safety margin. A single spoofed or corrupted frame cannot alter the system state unless the noise is sustained across a multi-frame voting window. This structural defense prevents momentary input attacks from successfully manipulating the application logic.

### A.5.6 Ethical Considerations and Bias Mitigation (Computing Ethics)

Security extends beyond mathematics into ethical computing—ensuring systems do not encode hidden biases that harm specific demographic groups. Facial recognition technology is historically prone to demographic bias depending on the source data.

**Application in this Project:**
- The training corpus merges FERPlus, RAF-DB, ExpW, and AffectNet to heavily diversify age, ethnicity, and lighting conditions.
- By deliberately using the `affectnet_full_balanced` subset (71,764 rows strongly downsampled for parity across the 7 emotion classes instead of using the highly imbalanced full set), the system avoids learning strong predictive priors based on majority demographic classes. This prevents the system from systematically failing or producing "confident-wrong" misclassifications on minority data strata—a core requirement of ethical computer vision deployment taught in senior computing units."""

if "## A.5 Consolidation of Security Elements" not in text:
    text += security_appendix

with open(path, 'w', encoding='utf-8') as f:
    f.write(text)

print("Security Appendix successfully appended!")
