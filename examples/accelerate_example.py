from tygent import accelerate


# Existing function representing some workflow
async def research_topic(topic: str):
    return {"summary": f"Research on {topic}"}


# Wrap the function with Tygent accelerate
accelerated_research = accelerate(research_topic)


if __name__ == "__main__":
    result = accelerated_research("AI trends")
    print(result)
