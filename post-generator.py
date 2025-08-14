import dspy

lm = dspy.LM("openai/gpt-4o", 
             max_tokens=200,
             temperature=1.3,
             cache=False)

system_prompt = """Given a prompt, generate a social media post that would produce engaging titles similar to those found on popular platforms under 200 tokens.
Examples include:

prompt: Generate a social media post against Trump.
title: Trump, 79, Has Deranged Rant about 'Grass' at Kennedy Center

prompt: Generate a social media post about the decline of PC gaming in Japan.
title: Japan's PC gaming population has decreased by 3 million in the past decade, studies suggest - AUTOMATON WEST

prompt: Generate a social media post about a funny incident at a bar.
title: I won a prize for dressing up for 80s night at the bar but I didn't know it was 80s night!

Return only the content of the post title, do not include any other text.



"""

prompts = []
prompt_post_dict = [] # [{"prompt": "prompt text", "posts": ["post1", "post2", ...]}]

# read generated_topics.txt file (new line delimited)
with open("generated_topics.txt", "r") as f:
    for line in f:
        prompts.append(line.strip())

# for each prompt, generate 10 posts
for prompt in prompts:
    posts = []
    for i in range(10):
        inputPrompt = system_prompt + prompt
        response = lm(inputPrompt)[0]
        posts.append(response.strip())

    prompt_post_dict.append({
        "prompt": prompt,
        "posts": posts
    })

# write the posts to a json file
import json
with open("generated_posts.json", "w") as f:
    json.dump(prompt_post_dict, f, indent=4)

