# Agentic Driving Coach

This repository contains the official implementation of the **Agentic Driving Coach**, a framework for ensuring robustness and determinism in agentic AI-powered human-in-the-loop (HITL) cyber-physical systems (CPS).

As foundation models and LLMs are increasingly adopted for complex reasoning in CPS, they introduce unpredictable behavior due to varying inference latencies and potential hallucinations. This project utilizes the **Reactor Model of Computation (MoC)** and the **Lingua Franca (LF)** framework to reintroduce determinism into these agentic systems.

---

## System Overview
The Agentic Driving Coach functions as an HITL CPS consisting of three key interacting components: an **Agentic Coach**, a **Human Driver**, and the **Physical Plant** (the car and its environment).

### Key Features

- **Deterministic Coordination**: Leveraging the Reactor MoC and the Lingua Franca (LF) framework to ensure consistent system behavior for a given set of inputs and initial states.
- **Deadline-Aware Inference**: Uses LF constructs to detect excessive LLM delays and trigger fallback safety mechanisms, such as emergency braking and pulling over.
- **Modal Planning**: Implements a modal model (monitoring, warning, and actuation) within the Planner reactor to moderate AI instructions provided to the human driver.
- **Structured Prompting**: Employs context-aware, zero-temperature prompts via Ollama to ensure precise, non-random control signals and instructions.

---

## Repository Structure
The project is organized around scenario-specific Lingua Franca programs:
- `StopSign.lf`: Evaluation of a car approaching a stop sign 100m ahead.
- `SpeedChange.lf`: Decelerating from 18 m/s to 11 m/s for a speed limit sign.
- `LaneChange.lf`: Managing a right-hand lane change while monitoring driver attention.

---

## Getting Started

### Prerequisites
1. **Lingua Franca (LF)**: Install the LF compiler and runtime from the [official website](https://www.lf-lang.org/).
2. **Ollama**: Install [Ollama](https://ollama.com/) to host local Llama 3 models.
3. **Llama 3 Models**: Pull the required 4-bit quantized models:
```
ollama pull llama3.2:1b
ollama pull llama3:8b
ollama pull llama3:70b
```

### Installation
Clone the repository and install the dependencies from your virtual environment:

```
git clone https://github.com/asu-kim/agentic-driving-coach.git
cd agentic-driving-coach
pip install -r requirements.txt
```

### Requirements.txt

```
torch==2.9.1
ollama==0.6.1
matplotlib==3.10.8
transformers==4.57.5
pandas==3.0.0
numpy==2.4.1
accelerate==1.12.0
pydantic==2.12.5
```
Lingua Franca installations for specfic systems can be found on the [installation page](https://www.lf-lang.org/docs/installation/).

---

## Contributors
- Deeksha Prahlad (dprahlad@asu.edu), Ph.D. student at Arizona State University
- Daniel Fan (danielfa@asu.edu), Undergraduate student at Arizona State University
- Hokeun Kim (hokeun@asu.edu, [https://hokeun.github.io/](https://hokeun.github.io/)), Assistant professor at Arizona State University
