# Farid Farahmand: Portfolio Knowledge Base

This document describes Farid Farahmand: his background, AI and software skills, experience, projects, education and interests. It is written to be used as the knowledge source for a portfolio chatbot, so each section is self-contained.

---

## 1. Overview

Farid Farahmand is a Software Engineer based in Montreal, Quebec, Canada, focused on AI engineering. He builds AI-powered applications with Retrieval-Augmented Generation (RAG), large language models (LLMs), vector databases and deep learning, and he backs them with solid software engineering in Python, TypeScript, Angular and React.

He holds a Master of Applied Computer Science from Concordia University, where he took graduate courses in Conversational AI, Natural Language Processing and machine learning. He has about three years of professional experience shipping production software, and he has since built two RAG systems (one with LangChain and ChromaDB, one from scratch with FAISS) and a deep learning model that detects Parkinson's disease from short speech recordings. He uses AI coding agents every day to write, review and test code.

## 2. Contact and Links

- Email: faridfarahmandf2f@gmail.com
- GitHub: github.com/FaridFarahmandf
- LinkedIn: linkedin.com/in/farid-farahmand
- Location: Montreal, Quebec, Canada

## 3. Professional Summary

Farid builds AI applications that are useful, reliable and grounded in real data. His strength is the combination of applied AI and production engineering: he understands how retrieval, embeddings and LLMs work, and he knows how to turn them into clean, tested applications that people can actually use. Before moving into AI, he spent three years as a software developer delivering enterprise and customer-facing applications, building production dashboards and real-time data visualization, and working directly with clients. That background means he thinks about AI systems as products, with attention to accuracy, user experience and maintainability.

## 4. AI Engineering Experience

This section summarizes Farid's hands-on AI work. Details for each project are in the Projects section.

**Retrieval-Augmented Generation (RAG)**
- Built two complete RAG systems: a RAG chatbot with a web interface (LangChain, ChromaDB, Streamlit, Llama 3.1) and a fully local RAG document Q&A pipeline built without a framework (sentence-transformers, FAISS, Ollama).
- Works with the full RAG pipeline: document ingestion (PDF, text, Markdown), chunking with overlap, embedding generation, vector indexing and similarity search, prompt construction with retrieved context, and LLM answer generation.
- Focuses on trustworthy answers: both systems ground responses in retrieved passages and cite their sources, which reduces hallucinations and lets users verify answers.
- Has worked with both local LLMs (Ollama, private and free to run) and cloud LLM APIs (Groq, faster inference), and understands the trade-offs between them in cost, speed and privacy.
- Is currently adding conversation history (multi-turn memory) to his RAG chatbot and plans to deploy it as a live demo.

**Deep learning and speech AI**
- Built a speech-based Parkinson's disease screening model using transfer learning from pretrained speech models (Wav2Vec 2.0, X-vector, ECAPA-TDNN) with PyTorch, SpeechBrain and Hugging Face.
- Designed experiments, including a multimodal pipeline combining phonation features with age signals, that lowered the error rate from 12.7% to 3.1% (a 76% relative improvement).

**AI-assisted engineering**
- Uses AI coding agents such as Claude daily to write, review and test code, and has delivered several projects with AI as a coding partner: he sets the goals and architecture, tests the output and iterates. He is open and transparent about this workflow.

**Academic foundation**
- Graduate courses in Conversational AI, Natural Language Processing and machine learning at Concordia University.
- Earned an A- in Conversational AI, taught by Professor Mirco Ravanelli, who has worked at Mila (Quebec AI Institute).
- Currently taking The AI Engineer Course: Complete AI Engineer Bootcamp.

## 5. Work Experience

### Software Developer, Sarmad Business Services Company
**August 2020 to September 2023** (about 3 years, starting as an intern)

Main responsibilities and achievements:

- Built a customer-facing e-commerce platform and four other enterprise applications using Angular, TypeScript and RxJS.
- Built production dashboards with NgRx state management and real-time data visualization.
- Created a shared library of 15+ reusable UI components (such as the menu and dashboard) that was adopted in 3 of the company's 5 applications.
- Reached about 70% automated test coverage using Jasmine, Karma and Cypress.
- Worked with Docker-based CI/CD pipelines.
- Used Vue 2 on two of the company's projects.
- Worked on the front end of a hospital patient intake and triage system, where patients enter their information and symptoms and the system routes them to the right department and doctor with relevant suggestions.

Notable projects and stories from this role:

- **Shared component library (his own initiative):** In his first year, Farid noticed that components like the menu and dashboard were being rebuilt from scratch on every project. He proposed a shared library. A senior developer initially pushed back because of a deadline, so they brought it to their manager together. The manager approved time to build it after that project, and the library ended up being used across most of the company's applications.
- **Travel tour agency application:** This was his last project at Sarmad, where he worked on the admin pages. After the first phase, his manager made him the direct client contact. He collected the client's bug reports and change requests, prioritized them (bugs first), reviewed feasibility with his manager, explained to the client which items would move to the next phase, and had the client test each release.
- **Pet shop application:** Right before release, testing showed that product search was slow, and fixing it required changes on both the front end and the back end. Farid showed the client the issue over Google Meet and offered two options: launch without search, or take a little more time. The client gave the team one more day, and the issue was fixed before release.

## 6. Education

### Master of Applied Computer Science, Concordia University, Montreal
**January 2024 to May 2026**

- Graduate courses in Conversational AI, Natural Language Processing and machine learning.
- Earned an A- in the Conversational AI course taught by Professor Mirco Ravanelli, who has worked at Mila (Quebec AI Institute). His Parkinson's disease speech detection project was built in this course.
- Also took a course that worked with image processing.

### Bachelor of Software Engineering, Azad University (Central Tehran Branch)
**September 2016 to June 2020**

## 7. Technical Skills

**AI and LLMs:** Retrieval-Augmented Generation (RAG), LangChain, prompt construction with retrieved context, source-cited answers, Llama 3.1, Ollama (local LLMs), Groq (cloud LLM API), Streamlit.

**Embeddings and vector search:** sentence-transformers, ChromaDB, FAISS, cosine similarity search, document chunking.

**Machine learning and deep learning:** PyTorch, Hugging Face, SpeechBrain, Wav2Vec 2.0, transfer learning, speech processing, model evaluation and experiment design.

**Programming languages:** Python, TypeScript, JavaScript (ES6+), Java, SQL.

**Back-end:** Flask, REST APIs, Node.js, JWT authentication, SQLAlchemy.

**Web development:** Angular (versions 19 to 21), React (React 19, hooks, Redux, Context API), RxJS, NgRx, Angular Material, Vue 2, HTML5, CSS3, SASS, Bootstrap, Chart.js.

**Databases:** PostgreSQL (via Supabase), MongoDB.

**Testing:** Jasmine, Karma, Cypress, Jest.

**DevOps and Tools:** Git, Docker, CI/CD, Vite.

**Cloud (fundamentals):** AWS (EC2, S3, Lambda) and Terraform at a fundamentals level, currently learning more.

**AI-assisted development:** Daily use of AI coding agents (such as Claude) to write, review and test code.

**Other:** Object-oriented design, design patterns, Agile/Scrum.

## 8. Projects

### RAG Chatbot with Web Interface
**GitHub:** github.com/FaridFarahmandf/chatbot-RAG
**Tech:** Python, LangChain, ChromaDB, Streamlit, Llama 3.1 (via Ollama and Groq)

A "chat with your documents" application. Users upload documents and ask questions, and the chatbot answers from the content of those documents, citing the source passages it used so every answer can be checked. Documents are split into chunks, embedded and stored in a ChromaDB vector database, and LangChain handles retrieval and passes the most relevant passages to the LLM. Users can switch between a local LLM through Ollama (private, no API costs) and a cloud LLM through Groq (faster responses). Farid is currently adding conversation history so the chatbot can handle follow-up questions, and he plans to deploy it as a live demo. This chatbot is part of his portfolio.

### RAG Document Q&A (Local RAG Pipeline Built from Scratch)
**GitHub:** github.com/FaridFarahmandf/rag-project
**Tech:** Python, sentence-transformers, FAISS, Ollama
**Built:** July to August 2026

A fully local document question-answering pipeline built without a framework, to understand every step of RAG directly. It ingests PDF, text and Markdown files, splits them into overlapping chunks, creates embeddings with sentence-transformers, stores them in a FAISS cosine-similarity vector index, retrieves the most relevant chunks for each question, and generates answers with a local LLM through Ollama. Answers are grounded in the retrieved passages and include source citations, with no external API costs and no data leaving the machine. The implementation was done with Claude as the coding assistant: Farid set the goals, tested and iterated.

### Detecting Parkinson's Disease from Short Speech Recordings
**GitHub:** github.com/FaridFarahmandf/Detecting-Parkinson-s-disease-from-short-speech-recordings
**Tech:** Python, PyTorch, SpeechBrain, Hugging Face, Wav2Vec 2.0
**Built:** January to April 2025, graduate project for the Conversational AI course at Concordia

A deep learning model that screens for Parkinson's disease from short speech recordings, using transfer learning from pretrained speech models. The three base models (X-vector, ECAPA-TDNN and Wav2Vec 2.0) and their starter code were provided by Professor Ravanelli. Farid's own work was the experiments built around them, including a multimodal pipeline that combined phonation features with age signals. That pipeline lowered the error rate from 12.7% to 3.1%, a 76% relative improvement.

### Finora: Stock Watchlist Tracker
**GitHub:** github.com/FaridFarahmandf/finora
**Tech:** React 19, Vite, Supabase (PostgreSQL), Clerk authentication, JWT, CSS3, Alpha Vantage REST API

A stock watchlist app where users sign in, save stocks and see color-coded daily percentage changes. It handles the Alpha Vantage API's rate limits with sequential, rate-limit-aware requests. Built with Claude as an AI coding assistant.

### Full-Stack Task Manager
**GitHub:** github.com/FaridFarahmandf/Task_Manager (front end) and github.com/FaridFarahmandf/Task_Manager_Backend (back end)
**Tech:** Angular 19+, TypeScript, Python, Flask, SQLAlchemy, JWT

A full-stack task management app. The Angular front end uses route guards, lazy loading, reactive forms, JWT authentication with expiration checking, and an HTTP interceptor that handles 401 responses. The back end is a Flask REST API built with Python and SQLAlchemy.

### Real-Time Market Analytics Dashboard
**GitHub:** github.com/FaridFarahmandf/Market-Analytics
**Tech:** Angular 19+, NgRx, RxJS, Chart.js, CoinGecko API

A real-time dashboard for cryptocurrency market data, with NgRx state management and live charts built with Chart.js.

### E-Commerce Store (React)
**GitHub:** github.com/FaridFarahmandf/store
**Tech:** React, JavaScript, CSS, Redux, Context API, Create React App, Jest

Originally the final project of a React course (around 2020 to 2021). In 2026, Farid modernized it and added a conversion feature set: a client-side A/B test on the add-to-cart button with sticky user bucketing, funnel tracking and a live results dashboard, a checkout upsell section ("You may also like"), and a "place order" conversion event. These additions were built with Claude as an AI coding assistant.

### Store Management REST API
**Tech:** Python, Flask

A REST API for managing stores and items, built with Flask.

### Global Dominion: Strategic Turn-Based Game Engine
**GitHub:** github.com/sayemomer/globalDominion
**Tech:** Java

A turn-based strategy game engine built as a team project, applying object-oriented design and design patterns.

## 9. Certifications and Courses

- In progress: The AI Engineer Course: Complete AI Engineer Bootcamp, Udemy
- Complete Python Developer, Zero to Hero, Udemy, summer 2025
- REST APIs with Flask and Python, Udemy (Jose Salvatierra), 2026
- Angular, Udemy (Mosh Hamedani), 2026
- React JS web development course, completed September 2026
- In progress: React, The Complete Guide (incl. Next.js, Redux), Udemy

## 10. Languages

- English: fluent
- French: intermediate, actively improving
- Persian (Farsi): native

## 11. Career Goals

Farid is looking for AI Engineer and Software Engineer roles in Montreal. He wants to build AI products: RAG systems, LLM-powered applications, AI assistants and tools that help people find and use information. He is especially interested in teams that ship AI to real users, where his mix of applied AI and production engineering adds the most value. He is open to new opportunities and can start right away.

## 12. Working Style

- Builds AI systems people can trust: prefers grounded, source-cited answers over impressive but unverifiable ones.
- Learns by building: wrote a RAG pipeline from scratch to understand each step before using frameworks.
- Takes initiative: proposed and built a shared component library to stop duplicated work.
- Communicates directly with clients and stakeholders, prioritizing issues and setting clear expectations.
- Values honest communication: when a problem appears before release, he presents the issue and the options rather than hiding it.
- Embraces AI-assisted development and is transparent about how he uses it.
- Comfortable in Agile teams.

## 13. Interests Outside of Work

- Photography, including portrait retouching.
- Tennis.

## 14. Frequently Asked Questions

**Who is Farid?**
Farid Farahmand is a Montreal-based Software Engineer focused on AI engineering. He builds RAG systems and LLM-powered applications, has a Master of Applied Computer Science from Concordia University, and has about three years of professional software experience.

**Is Farid a good fit for an AI Engineer role?**
Yes. He has built two complete RAG systems, a deep learning speech model with strong results, and he studied Conversational AI, NLP and machine learning at the graduate level. He also brings three years of production software experience, so he can take an AI idea all the way to a working, tested application.

**What AI experience does Farid have?**
He built a RAG chatbot with LangChain, ChromaDB, Streamlit and Llama 3.1; a local RAG Q&A pipeline from scratch with sentence-transformers, FAISS and Ollama; and a Parkinson's disease detection model from speech using PyTorch, SpeechBrain and Wav2Vec 2.0. He took graduate courses in Conversational AI (A-), NLP and machine learning at Concordia, and he uses AI coding agents every day.

**What does Farid know about RAG?**
He understands and has implemented the full pipeline: document ingestion, chunking with overlap, embeddings with sentence-transformers, vector storage and similarity search with ChromaDB and FAISS, retrieval with LangChain, and grounded answer generation with source citations. He has built it both with a framework (LangChain) and without one (from scratch).

**Which LLMs and AI tools has Farid used?**
Llama 3.1 through Ollama (local) and Groq (cloud), LangChain, ChromaDB, FAISS, sentence-transformers, Hugging Face, PyTorch and SpeechBrain. He uses Claude daily as an AI coding agent.

**Has Farid trained or fine-tuned models?**
In his Parkinson's project he worked with pretrained speech models (Wav2Vec 2.0, X-vector, ECAPA-TDNN) through transfer learning and ran the experiments, including a multimodal pipeline that cut the error rate from 12.7% to 3.1%. The base models and starter code were provided by his professor.

**What is his strongest project?**
His RAG Chatbot is his flagship AI project, and it is the technology behind this chatbot. The Parkinson's detection project shows his deep learning skills, and the local RAG pipeline shows he understands RAG from the ground up.

**What is he working on now?**
He is adding conversation history to his RAG chatbot so it can handle follow-up questions, and preparing to deploy it as a live demo. He is also taking an AI Engineer bootcamp.

**How many years of experience does Farid have?**
About three years of professional software development (August 2020 to September 2023), followed by a Master's degree with graduate AI courses and several AI projects.

**Does Farid know Python?**
Yes. Python is his main language for AI work (RAG systems, PyTorch, Hugging Face) and for back-end development with Flask.

**Can Farid build the application around an AI model?**
Yes. He has three years of experience building production web applications with Angular and TypeScript, and he also builds with React, Flask and REST APIs, so he can deliver the user interface, API and AI components together.

**Does Farid know cloud technologies?**
He knows AWS (EC2, S3, Lambda) and Terraform at a fundamentals level and is continuing to learn.

**Where is Farid located?**
Montreal, Quebec, Canada.

**Does Farid speak French?**
Yes, at an intermediate level, and he is actively improving it.

**How can I contact Farid?**
By email at faridfarahmandf2f@gmail.com, or on LinkedIn at linkedin.com/in/farid-farahmand. His code is on GitHub at github.com/FaridFarahmandf.
