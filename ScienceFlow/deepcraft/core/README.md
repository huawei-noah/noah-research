# core

* Builds the most fundamental core functionality and data structure abstractions. It can be compiled and modified independently, and optimized on its own as long as the interfaces stay consistent.
* The Core components can build basic LLM applications: dialogue, memory, RAG, serial/parallel tool calling, etc. (examples/01~04xx.py).


### Directory tree
```
├─llm (currently supports online models, using the openai-api as the interface)
├─memory (depends on Message; builds the basic memory system)
├─message
└─tool
```
