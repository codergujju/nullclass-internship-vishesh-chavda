import os
import xml.etree.ElementTree as ET
import csv

DATA_DIR = "Dataset"

CSV_FILE = "medical_qa_data.csv"

with open(CSV_FILE, mode="w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Tag", "Question", "Answer"])  

    for folder in os.listdir(DATA_DIR):
        folder_path = os.path.join(DATA_DIR, folder)

        if os.path.isdir(folder_path):
            tag = folder.lower()
            for file in os.listdir(folder_path):
                if file.endswith(".xml"):
                    file_path = os.path.join(folder_path, file)

                    try:
                        tree = ET.parse(file_path)
                        root = tree.getroot()

                        question = root.findtext(".//Question")
                        answer = root.findtext(".//Answer")

                        if question and answer:
                            writer.writerow([tag, question.strip(), answer.strip()])

                    except Exception as e:
                        print(f" Error in {file_path}: {e}")

print(f" XML data successfully converted to: {CSV_FILE}")
