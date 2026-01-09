
# Secure Financial AI Agent 

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-Agentic-orange)
![Security](https://img.shields.io/badge/Security-Guardrails-red)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)

## Overview

This project implements a **secure, autonomous AI agent** designed specifically for the financial sector constraints. Unlike standard chatbots, this system uses an **Agentic Workflow** to intelligently route user queries between **Knowledge Retrieval (RAG)** and **Operational Tasks (Tools)**.

Crucially, it integrates a **Security Layer (Guardrails)** to prevent adversarial attacks and enforce financial compliance, alongside a **Continuous Evaluation Pipeline**.

## Architecture

1.  **Input Guardrails:** Filters malicious prompts and off-topic queries before they reach the model.
2.  **Router/Planner:** Decides the necessary step:
    * *Retrieval:* Querying the Vector Database.
    * *Action:* Executing a defined tool.
    * *Direct Answer:* Handling chitchat securely.
3.  **Output Guardrails:** Verifies the generated response for factual consistency and compliance.
4.  **Auto-Evaluation:** A background pipeline scores interaction quality.


