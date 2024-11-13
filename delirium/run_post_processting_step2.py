import csv
import glob
import os
import re

import xml.etree.ElementTree as ET


def read_file_list(indir, d):
    opt_notes = []
    with open(indir, "r") as csvfile:
        spamreader = csv.reader(csvfile, delimiter=d)
        for row in spamreader:
            opt_notes += [row]
    return opt_notes


def read_txt(indir):
    f = open(indir, "r")
    txt = f.read()
    f.close()
    return txt


def is_file_size_greater_than_zero(file_path):
    try:
        file_size = os.path.getsize(file_path)
        return file_size > 0
    except OSError:
        # If the file doesn't exist or there's an error accessing it
        return False


def cam_cp(cp):
    cp = cp.strip()
    cam = ""
    if cp in ["encephalopathy", "agit", "ams", "fluctu", "confusion"]:
        cam = "cam_a"
    elif cp in ["inattention"]:
        cam = "cam_b"
    elif cp in ["hallucination", "disorganized_thinking", "disorient"]:
        cam = "cam_c"
    elif cp in ["disconnected"]:
        cam = "cam_d"
    elif cp == "delirium":
        cam = "delirium"
    return cam


def cam_original(cam_list):
    cam_list = [i.lower() for i in cam_list if i != ""]
    delirium_status = 0
    if (
        "cam_a" in cam_list
        and "cam_b" in cam_list
        and "cam_d" in cam_list
        and "cam_c" in cam_list
    ):
        delirium_status = 1
    elif "cam_a" in cam_list and "cam_b" in cam_list and "cam_d" in cam_list:
        delirium_status = 1
    elif "cam_a" in cam_list and "cam_b" in cam_list and "cam_c" in cam_list:
        delirium_status = 1
    elif "delirium" in cam_list or "deli_cam" in cam_list:
        delirium_status = 1
    elif cam_list != []:
        delirium_status = 0
    return delirium_status


def output_evidence(indir):
    nlp_patient_norm, nlp_patient_cam = {}, {}
    nlp_result_txt = glob.glob(indir)

    for i in nlp_result_txt:
        note = read_file_list(i, "\t")
        if note == []:
            continue
        for row in note:
            certainty = row[6]
            status = row[7]
            experiencer = row[8]
            if "Positive" in certainty and "Patient" in experiencer:
                norm = row[9].lower().split('"')[1]
                cam = cam_cp(norm)
                file_name = row[0]
                if file_name not in nlp_patient_norm:
                    nlp_patient_norm[file_name] = [norm]
                else:
                    nlp_patient_norm[file_name] = nlp_patient_norm[file_name] + [norm]
                if file_name not in nlp_patient_cam:
                    nlp_patient_cam[file_name] = [cam]
                else:
                    nlp_patient_cam[file_name] = nlp_patient_cam[file_name] + [cam]

    with open("./summarized_result.csv", "w") as csvfile:  # store in the current dir
        spamwriter = csv.writer(csvfile, delimiter="|")
        for file_name in nlp_patient_cam:
            nlpcam = cam_original(nlp_patient_cam[file_name])
            spamwriter.writerow(
                [
                    file_name,
                    nlp_patient_norm[file_name],
                    nlp_patient_cam[file_name],
                    nlpcam,
                ]
            )


def create_combined_xml(raw_notes, nlp_output, model_name):
    # Create the root element
    root = ET.Element(model_name)

    # Add the TEXT section
    text_element = ET.SubElement(root, "TEXT")
    text_element.text = f"<![CDATA[{raw_notes}]]>"

    # Add the TAGS section
    tags_element = ET.SubElement(root, "TAGS")

    # Parse NLP output and create corresponding XML tags
    for line in nlp_output.strip().splitlines():

        parts = line.split("\t")
        del parts[1]
        tag_attributes = {
            "spans": "{}~{}".format(
                parts[3].split("=")[1].strip('"'), parts[4].split("=")[1].strip('"')
            ),
            "id": parts[2].split("=")[1].strip('"')[:2].upper()
            + str(nlp_output.strip().splitlines().index(line)),
            "certainty": parts[5].split("=")[1].strip('"').lower(),
            "status": parts[6].split("=")[1].strip('"').lower(),
            "experiencer": parts[7].split("=")[1].strip('"').lower(),
            "text": parts[2].split("=")[1].strip('"'),
            "exclusion": "",
            "comment": "",
        }

        # Replace spaces and special characters with underscores
        tag_name = parts[8].split("=")[1].strip('"').capitalize().replace(" ", "_")
        tag_name = re.sub(
            r"[^A-Za-z0-9_]", "_", tag_name
        )

        if not re.match("^[A-Za-z_][A-Za-z0-9_.-]*$", tag_name):
            raise ValueError(f"Invalid tag name '{tag_name}' in line: {line}")

        ET.SubElement(tags_element, tag_name, tag_attributes)

    xml_str = ET.tostring(root, encoding="utf-8", method="xml").decode("utf-8")
    xml_str = xml_str.replace("&lt;![CDATA[", "<![CDATA[").replace("]]&gt;", "]]>")
    xml_str = f'<?xml version="1.0" encoding="UTF-8" ?>\n{xml_str}'
    return xml_str


def prepare_medtator_annotation(txt_dir, ann_dir, model_name):
    txt = glob.glob(txt_dir)
    nlp = glob.glob(ann_dir)
    for i in nlp:
        if is_file_size_greater_than_zero(i):
            raw_notes = read_txt(i.replace(".ann", "").replace("output", "input"))
            nlp_output = read_txt(i)
            combined_xml = create_combined_xml(raw_notes, nlp_output, model_name)
            outdir = i.replace(".ann", "").replace("output", "medtator") + ".xml"
            os.makedirs(os.path.dirname(outdir), exist_ok=True)
            print(outdir)
            with open(outdir, "w") as text_file:
                text_file.write(combined_xml)



def main():
    # specify MedTagger output folder director
    MODEL_NAME = "Delirium_schema_1_3"
    raw_txt_dir = "/delirium/data/input"
    nlp_ann_dir = "/delirium/data/output"
    output_evidence(nlp_ann_dir + "/*.ann")
    prepare_medtator_annotation(raw_txt_dir + "/*.txt", nlp_ann_dir + "/*.ann", 
    model_name=MODEL_NAME)


if __name__ == "__main__":
    main()
