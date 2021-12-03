#!/usr/bin/env python
# coding: utf-8

from xml.dom.minidom import parseString
import re
import os
import sys
import string


def process_wiki_file(input_file_path, output_dir):
    article_min_word_limit = 64

    digit_name = os.path.split(input_file_path)[1].split("_")[1]
    with open(input_file_path, "r") as xml_file:
        xml_data_str = xml_file.read()

    output_file_dir = os.path.join(output_dir, digit_name)
    os.makedirs(output_file_dir, exist_ok=True)
    
    xml_data_str = xml_data_str.replace("&g.", "")
    root = parseString("<dummyroot>" + xml_data_str + "</dummyroot>").documentElement

    exclude_punctuation = set(string.punctuation)
    
    for doc in root.childNodes:
        if doc.toxml().strip() == "":
            continue

        doc_id = int(doc.getAttribute("id"))
        doc_title = doc.getAttribute("title")
        doc_url = doc.getAttribute("url")
        doc_text = doc.firstChild.nodeValue
        word_count = len(re.findall(r"\w+", doc_text))    
        if word_count < article_min_word_limit:
            continue

        doc_lines = [line.strip() for line in doc_text.split("\n") if line.strip() != ""]
        doc_lines[0] = "####2 Main." 

        doc_clean_lines = []
        topic_level2 = "####2"
        for line in doc_lines:
            if line.startswith("####"):
                if not line.startswith(topic_level2 + " ") or line.strip() == topic_level2:
                    continue
            doc_clean_lines.append(line)

        topic_line_list = []    
        i = 0
        while i < len(doc_clean_lines):
            line = doc_clean_lines[i]
            if line.startswith("####2"):
                topic_line_list.append(i)
            i += 1
        topic_line_list.append(i)

        for i in range(len(topic_line_list) - 1):
            start = topic_line_list[i]
            end = topic_line_list[i + 1]
            topic_text = "".join(doc_clean_lines[start + 1:end])
            topic_name = doc_clean_lines[start].split(" ", 1)[1].lower()            
            topic_name = ''.join(ch for ch in topic_name if ch not in exclude_punctuation)
            topic_name = topic_name.replace(" ", "_")

            sample_file_path = os.path.join(output_file_dir, "%s_%09d_%02d" % ("topic", doc_id, i))
            with open(sample_file_path, "w") as sample_file:
                sample_file.write('%s;%d;"%s";"%s"\n' % (topic_name, doc_id, doc_title, doc_url))
                sample_file.write('%s\n' % (topic_text))

def main():
    if len(sys.argv) != 3:
        print("Incorrect number of arguments. wiki_topic_exctractor.py input_dump_text output_topic_path")
        return 1

    base_input_path = sys.argv[1]
    base_output_path = sys.argv[2]

    all_in_file_paths = sorted([os.path.join(dp, f) for dp, dn, filenames in os.walk(base_input_path) for f in filenames])
    for in_file_path in all_in_file_paths:
        print("Progress: " + in_file_path)
        base_subpath = in_file_path.split(os.path.sep)[-2]
        process_wiki_file(in_file_path, os.path.join(base_output_path, base_subpath))

    return 0

sys.exit(main())




