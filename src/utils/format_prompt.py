def format_prompt(text: str, task: str) -> str:
    """
    Wrap the input text in a task-specific instruction prompt.

    Parameters
    ----------
    text : str
        The user-provided content to classify.
    task : str
        The classification task identifier (e.g., 'toxicity', 'hate', 'offense').

    Returns
    -------
    str
        The formatted prompt string ending with a 'Label:' indicator.
    """
    # Simple instruction + user text -> ensures final token is "Label:" to predict after
    instruction = {
        "toxicity": "Decide if the following text is toxic or non-toxic. Answer with a single word.\nText: ",
        "hate": "Decide if the text is hateful or not hateful. Answer with a single word.\nText: ",
        "offense": "Decide if the text is offensive or not offensive. Answer with a single word.\nText: ",
    }[task]
    return f"{instruction}{text}\nLabel:"