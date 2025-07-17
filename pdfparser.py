import pdfplumber

def read_pdf_with_tables(file_path):
    full_text = []

    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            # Extract text
            text = page.extract_text()
            if text:
                full_text.append(f"--- Page {i + 1} Text ---\n{text.strip()}")

            # Extract tables
            tables = page.extract_tables()
            for t_index, table in enumerate(tables):
                full_text.append(f"\n--- Page {i + 1} Table {t_index + 1} ---")
                for row in table:
                    row_text = '\t'.join(cell.strip() if cell else '' for cell in row)
                    full_text.append(row_text)

    return '\n'.join(full_text)

# # Example usage
# if __name__ == "__main__":
#     file_path = "Non-Binding Term Sheet Sample.pdf"  # Replace with your actual PDF file
#     content = read_pdf_with_tables(file_path)
#     print("Extracted PDF Content with Tables:\n")
#     print(len(content))
