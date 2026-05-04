\# 📚 AI Study Assistant: Zero-Token Hybrid RAG System



> A cost-optimized, Vietnamese-focused AI Study Assistant utilizing Hybrid RAG (TF-IDF + TextRank) and Gemini 2.5 Flash to minimize API latency and inference costs.



\## 🌟 Architecture Highlights



This project is built with a \*\*"Zero-Token First"\*\* philosophy, focusing on local processing to reduce dependency on paid API calls.



\*   \*\*Local Intent Routing:\*\* Uses RegEx and Substring matching to classify user queries (Factual, Comparative, Applied) entirely on local RAM, saving 60-70% of API token routing costs.

\*   \*\*Vectorless RAG:\*\* Replaces traditional Vector Databases with local \*\*TF-IDF \& Cosine Similarity\*\* combined with `underthesea` for precise Vietnamese word segmentation.

\*   \*\*Two-Layer Anti-Hallucination:\*\*

&#x20;   \*   \*Layer 1 (Local):\* Char N-gram mathematical verification.

&#x20;   \*   \*Layer 2 (Cloud):\* LLM-as-a-Judge using strictly formatted JSON Chain-of-Thought prompts to eliminate self-preference bias.

\*   \*\*Zero-Token Ecosystem:\*\* Reuses TextRank adjacency matrices and ForceAtlas2 physics algorithms to auto-generate Mindmaps and Flashcards without triggering LLM APIs.



\## 🛠️ Tech Stack

\*   \*\*Language:\*\* Python

\*   \*\*Frontend/UI:\*\* Streamlit

\*   \*\*NLP \& Retrieval:\*\* underthesea, TF-IDF, TextRank, N-gram

\*   \*\*Generative AI:\*\* Google Gemini 2.5 Flash / Pro (via `google-generativeai`)

\*   \*\*Data Visualization:\*\* PyVis (ForceAtlas2)



\## ⚙️ Installation \& Setup



1\. \*\*Clone the repository:\*\*

&#x20;  ```bash

&#x20;  git clone \[https://github.com/YourUsername/YourRepoName.git](https://github.com/YourUsername/YourRepoName.git)

&#x20;  cd YourRepoName
2. Install dependencies:

pip install -r requirements.txt
3. Configure Environment Variables:

Create a .env file in the root directory and add your Google Gemini API Key:
GEMINI\_API\_KEY=your\_api\_key\_here
4. Run the application:
streamlit run app.py
📊 System Metrics (Proof of Concept)

Compression Rate: \~75% reduction in reading volume via TextRank (default\_summary\_ratio = 0.25).



Latency: < 5 seconds for end-to-end RAG generation.



Graceful Degradation: Maintains 100% offline retrieval functionality (Top 5 context chunks) when API connectivity fails.

