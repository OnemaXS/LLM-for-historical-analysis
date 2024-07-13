from langchain.chat_models import ChatOpenAI
from langchain.schema import Document


def get_variable_name(variable):
    """Returns name of a variable as a str. 
    Adapted from: https://www.geeksforgeeks.org/how-to-print-a-variables-name-in-python/"""
    variable_name = [name for name, value in globals().items() if value is variable][0]
    return variable_name

def initialize_retriever_older(vectorstore, k = 7):
    var_name = get_variable_name(vectorstore)
    print(f"Initialized the retriever out of {var_name}. {k} documents will be retrieved. ")
    return vectorstore.as_retriever(search_kwargs={"k": k}), (var_name, k)

def initialize_retriever(vectorstore, k = 7):
    """Initializes a simple similarity-search retriever from the vectorstore, set to retrieve k documents.
    Returns the retriever and a tuple with its hyperparameters for record.""" 
    var_name = input("Which vector database are you using? Please provide its name for record: ")
    print(f"Initialized the retriever out of {var_name}. {k} documents will be retrieved.")
    return vectorstore.as_retriever(search_kwargs={"k": k}), (var_name, k)

def initialize_llm(model_name = "gpt-3.5-turbo", temp = 0.2, max_tokens = 400):
    
    """Creates an LLM and reports whether the hyperparameters are standard for the project.
    Keeps all hyperparameters defined only locally. Returns the llm and a tuple containing its hyperparameters for record. """
    
    standard = ["gpt-3.5-turbo", 0.2, 400]  # standard values, as chosen during hyperparameter selection
    passed = [("model name", model_name), ("temp", temp), ("max_tokens", max_tokens)]  # names and values of passed args
    is_standard = [standard[i] == p[1] for i, p in enumerate(passed)]  # boolean list checking whether passed args are standard
    
    if all(is_standard):
        print("Created an LLM with standard hyperparameters.")
    
    else:
        print("WARNING! Created an LLM with non-standard hyperparameters:")
        for i, boolean_statement in enumerate(is_standard):
            if not boolean_statement:
                print(f"{passed[i][0]} has a non-standard value: {passed[i][1]}")  
        
    return ChatOpenAI(model_name=model_name, temperature=temp, max_tokens=max_tokens), (model_name, temp, max_tokens)
    
def pretty_metadata(metadata_dict):
    """Prints metadata in a citation-like format."""
    return metadata_dict["autor"]+", "+metadata_dict["source"]+", ed. "+str(metadata_dict["edition"])+" ("+str(metadata_dict["date"])+"), "+metadata_dict["in-text location"]

def format_docs(docs):  # ispired by: https://python.langchain.com/docs/use_cases/question_answering/sources 
    """Prints the retrieved documents and reformats them before passing to the LLM. """   
    returned_sources = []
    print("Retrieved context (metadata are not passed to the model):\n")
    for doc in docs:
        print(doc.page_content+"\n"+"Metadata: "+pretty_metadata(doc.metadata)+"\n")
        returned_sources.append(doc.page_content)
    return "\n\n".join(returned_sources)

def print_prompt(prompt):
    """Prints the whole prompt."""
    print("The prompt as passed to the model:\n")
    print(prompt)
    return(prompt)   

def format_docs_for_documentation(docs):
    """Formats the documents for saving, e.g. in a txt file."""
    returned_sources = []
    for doc in docs:
        returned_sources.append(doc.page_content+"\n"+"Metadata: "+pretty_metadata(doc.metadata)+"\n")
    return "\n".join(returned_sources)

def format_docs_with_metadata(docs):
    """Similar to format_docs, however, the metadata are not only printed out, but also passed to the model."""
    returned_sources = []
    for doc in docs:
        formatted_doc = doc.page_content+"\n"+"Metadata: "+pretty_metadata(doc.metadata)+"\n"
        print(formatted_doc)
        returned_sources.append(formatted_doc)
    return "\n".join(returned_sources)
