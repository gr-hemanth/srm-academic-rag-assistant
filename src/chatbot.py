from rag import get_answer

print("\nSRM Academic Assistant")
print("Type 'exit' to quit.\n")

while True:

    query = input("You: ").strip()

    if query.lower() == "exit":
        print("\nGoodbye!")
        break

    answer, sources = get_answer(query)

    print("\nAssistant:")
    print(answer)

    print("\nSources:")

    for source_pdf, page in sources:
        print(f"{source_pdf} - Page {page}")

    print()