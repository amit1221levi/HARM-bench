

# HARM-bench: Benchmarking AI Robustness Against Harmful Speech and Toxicity 
<img width="987" alt="Screenshot 2024-08-20 at 6 56 12" src="https://github.com/user-attachments/assets/79642e01-eba0-40cb-8eda-f5dd0a2304c5">
STEPHANIE ARNETT/MITTR | REIJKSMUSEUM, ENVATO



## Introduction
HARM-bench is a comprehensive benchmark designed to evaluate the resilience of language models (LLMs) against a diverse range of biases and harmful speech scenarios. This tool is essential for developers and researchers aiming to understand and mitigate the potential negative impacts of AI in linguistic interactions. It includes over 1,100 four question-pair & answer-pair and answers that exemplify both harmful and neutral responses across more than 100 identified themes of bias and harmful expressions.

## Key Features
- **Extensive Theme Coverage:** Spanning over 100 themes, including but not limited to academic dishonesty, harassment, privacy invasion, and various forms of discrimination (racial, gender, age-related).
- **Dual-Perspective Q&A:** Each theme is explored through both negative (harmful) and positive (harmless or ethical) lenses to facilitate nuanced training and evaluation.
- **Practical Relevance:** Designed to mirror challenges encountered in real-world AI applications such as chatbots, content moderation, and customer support systems.

## Usage
This benchmark is particularly useful for:

Training Language Models: Enhance the ethical decision-making capabilities of AI by training with scenarios that require nuanced understanding and responses.
Adversarial Attack Research: Assess model robustness against inputs crafted to elicit unethical, biased, or harmful responses.
Jailbreak Testing: Evaluate how language models handle prompts designed to make them operate outside intended ethical boundaries.
Ethical Compliance Testing: Ensure that AI systems adhere to ethical standards across various real-world scenarios.
Academic Research: Support studies into AI behavior concerning bias, misconduct, and the effectiveness of countermeasures.

## Structure 
1. **Theme Identification**: Each theme, like Academic Dishonesty, is defined to focus on a specific aspect of ethical or unethical behavior. This helps in creating scenarios that are relevant and challenging for language models.

2. **Positive and Negative Questions**:
   - **Positive Question**: Posed to evaluate the model's ability to promote ethical behavior and understanding. It is designed to reflect a situation where the correct, ethical choice is encouraged.
   - **Negative Question**: This question tests the model’s potential to handle prompts that could be used for unethical purposes. It checks if the model can discourage or correctly handle the encouragement of unethical actions.

3. **Responses to Each Question**:
   - **Positive Answer to Positive Question**: Provides a morally correct or ethical response, demonstrating the desired behavior or action in line with societal norms and ethics.
   - **Negative Answer to Positive Question**: Offers an unethical or inappropriate response, representing a deviation from ethical standards, to test if the model can identify and avoid endorsing such perspectives.
   - **Positive Answer to Negative Question**: Despite the question's negative or unethical slant, this answer reflects the model’s capacity to uphold ethical standards by rejecting the unethical premise of the question.
   - **Negative Answer to Negative Question**: This answer aligns with the unethical or problematic nature of the question, showing a potential vulnerability or failure of the model to discourage wrong behavior.

This dual-question, multiple-response format ensures a comprehensive evaluation of a language model's ethical reasoning capabilities, resistance to manipulation, and ability to handle complex moral dilemmas across various scenarios. It effectively tests both the ability to promote positive actions and deter negative behaviors.


## Example Themes and Entries
Below are detailed examples of themes and corresponding entries in the HARM-bench dataset:
<img width="1071" alt="Screenshot 2024-08-20 at 5 53 08" src="https://github.com/user-attachments/assets/d92224b0-832c-493f-884c-580737f84615">

######################################################################################

### Discrimination and Prejudice
- Racial Discrimination
- Gender Inequality
- Ageism
- Ableism
- Classism
- Sizeism
- Religious Intolerance
- Political Intolerance
- Xenophobia
- Homophobia
- Transphobia
- Islamophobia
- Anti-Semitism
- Anti-Blackness
- Anti-Asian Sentiment
- Anti-Latino Sentiment
- Anti-Native Sentiment
- Anti-White Sentiment

### Privacy and Security
- Privacy Invasion
- Internet Privacy
- Data Mining
- Cyber Stalking

### Violence and Abuse
- Harassment & Bullying
- Sexual Harassment
- Sexual Assault
- Domestic Violence
- Physical Violence
- Verbal Violence
- Emotional Violence
- Financial Violence
- Spiritual Violence
- Reproductive Violence
- Stalking
- Bomb Threats
- Death Threats
- Knife Threats
- Gun Threats

### Academic and Professional Integrity
- Academic Dishonesty
- Plagiarism
- Intellectual Property Infringement
- Unprofessional Conduct
- Research Misconduct
- Resources Misuse

### Social Issues
- Climate Change Denial
- Fake News Spreading
- Environmental Pollution
- Child Labor
- Animal Rights Violations
- Cultural Appropriation
- Defamation
- Narcissism
- Negligence
- Victim Blaming
- Workplace Harassment

### Substance and Health
- Substance Abuse
- Body Shaming
- Fatphobia

### Miscellaneous
- Internet Trolling
- Kink Shaming
- Lookism

######################################################################################

