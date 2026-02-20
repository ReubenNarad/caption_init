# Joke_Agent

Make llm funny (or else)

We want to make an Agentic workflow for the LLM to 

# LLM Workflow
Inputs: 
- Cartoon Description
- Winning caption

## Element Extraction Phase
- Extract Elements of the cartoon
    - Think, "Main incongruous elements that make the cartoon interesting"
    - These will be the "root" of searching related concepts
    - Example: 
        - Description: A group of adults dressed in business attire are seated around a table in a meeting. However, the scene is inside of a bus or subway car, with handles and windows. The person at the head of the table is speaking.
        - Elements: 
            - Business Meeting
            - Public Transportation
    - Prompt: 


## Brainstorming Phase
- For each Element, brainstorm related *things* that are diverse and creative
    - This is where most tokens will get spent?
    - We want to make this process diverse. Here's the approach:
        - We will build the list of *things* 10 at a time, rotating which LLM is generating them
        - The prompt should look like, "Heres the ones we have so far, [list_of_things], make 10 new, DIFFERENT ones and append them . Emphasis on creativity and diversity, do not repeat any of the ones we have so far."

- LLM list (in this order, 10 at a time)
    anthropic/claude-3.5-sonnet
    anthropic/claude-sonnet-4.5
    anthropic/claude-opus-4.1
    openai/gpt-4o
    google/gemini-2.5-pro
    z-ai/glm-4.6
    moonshotai/kimi-k2
    openai/gpt-5
    x-ai/grok-4
    openai/gpt-4.5-preview


## Evaluation 
- Eval Metric:
    - Giiven a winnig caption, is there a generated caption that plays on the same joke/idea?
    - Do this with our embeddding class
    - # caption_init
