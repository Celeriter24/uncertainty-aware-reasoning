# uncertainty-aware-reasoning
The idea is that instead of sampling the most probable next token by an LLM, to compare beforehand the logits of its answer with some "uncertain" answers. If there is uncertainty in its first answer, it should be able to communicate what it is uncertain about towards the user to improve the overall capability of the LLM.
