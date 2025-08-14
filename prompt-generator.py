import dspy

lm = dspy.LM("openai/gpt-4o", 
             max_tokens=200,
             temperature=1.3,
             cache=False)

prompt = """Write a prompt about a specific topic that would generate titles like those found on social media under 200 tokens.
Examples include:

prompt: Generate a social media post against Trump.
title: Trump, 79, Has Deranged Rant about 'Grass' at Kennedy Center

prompt: Generate a social media post about the decline of PC gaming in Japan.
title: Japan's PC gaming population has decreased by 3 million in the past decade, studies suggest - AUTOMATON WEST

prompt: Generate a social media post about a funny incident at a bar.
title: I won a prize for dressing up for 80s night at the bar but I didn't know it was 80s night!

Return only the prompt, do not include any other text.
"""

topics = []
for i in range(20):
    response = lm(prompt)[0]
    topics.append(response)
    print(f"Generated Topic {i+1}: {response.strip()}")

print("Generated Topics:")
for topic in topics:
    print(topic.strip())

# Save the generated topics to a file
with open("generated_topics.txt", "w") as f:
    for topic in topics:
        f.write(topic.strip() + "\n")