# Agentic Driving Coach 

This project implements a real-time agentic driving coach using Lingua Franca (LF) for deterministic execution and MuJoCo for physics-based vehicle simulation. It integrates driver inputs, LLM-based guidance and MuJoCo simulation of the car.


## Submodule Initialization

git submodule update --init --recursive 


## Prerequisites
1. **Lingua Franca (LF)**: Install the LF compiler and runtime from the [installation page](https://www.lf-lang.org/docs/installation/).
2. **Ollama**: Install [Ollama](https://ollama.com/) to host local Llama 3 models.
3. **Llama 3 Models**: Pull the required 4-bit quantized models:
```
ollama pull llama3.2:1b
ollama pull llama3:8b
ollama pull llama3:70b
```
4. For MuJoCo:
```
python3 -m pip install mujoco
```

## Build the LF program

lfc src/HardwareIntegration/HardwareIntegratedStopSign.lf   

### Run

For federated execution:

```
cd ~/agentic-driving-coach/fed-gen/HardwareIntegratedStopSign/
```
In separate terminals run:
1. On the server:
```
./bin/RTI -n 3 
```
2. On the Raspberry Pi:
```
./bin/federate__c  
```
3. On the server:
```
./bin/federate__en
```
4. On the server:
```
./bin/federate__adc 
```



## Original MuJoCo LF Program 

Edward A. Lee, [Email](eal@berkeley.edu)
Link: [MuJoCo Py](https://github.com/lf-lang/mujoco-py)


