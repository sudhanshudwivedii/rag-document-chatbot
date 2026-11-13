from langchain_community.document_loaders import TextLoader

data=TextLoader("data.txt")

docs=data.load()
print(docs)