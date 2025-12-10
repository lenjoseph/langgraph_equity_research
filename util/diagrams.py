def draw_architecture(graph_workflow):
    try:
        png_data = graph_workflow.get_graph().draw_mermaid_png()
        with open("architecture.png", "wb") as f:
            f.write(png_data)
    except Exception as e:
        print(f"Error generating architecture.png: {e}")
        # Fallback to writing mermaid text
        with open("architecture.mmd", "w") as f:
            f.write(graph_workflow.get_graph().draw_mermaid())
