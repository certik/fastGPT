from time import monotonic as clock
import os; os.environ["OMP_NUM_THREADS"] = "1"
print("Importing")
t1 = clock()
from transformers import pipeline
t2 = clock()
print("  Time: ", t2-t1)
print("Loading")
t1 = clock()
generator = pipeline('text-generation', model='gpt2')
t2 = clock()
print("  Time: ", t2-t1)
text="Alan Turing theorized that computers would one day become very powerful, but even he could not imagine"
print("Generating")
t1 = clock()
g = generator(text, do_sample=False, max_new_tokens=20)
t2 = clock()
print("  Time: ", t2-t1)
output = g[0]["generated_text"]
print(output)
