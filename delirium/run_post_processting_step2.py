import csv
import glob
import os

from lxml import etree


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
    print("loading finished")

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

    with open("data/summarized_result.csv", "w") as csvfile:
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


def create_combined_xml(raw_notes, nlp_output):
    # Create the root element
    root = etree.Element("Delirium_schema_1_3")

    # Add the TEXT section with CDATA
    text_element = etree.SubElement(root, "TEXT")
    text_element.text = etree.CDATA(raw_notes)

    # Add the TAGS section
    tags_element = etree.SubElement(root, "TAGS")

    # Parse NLP output and create corresponding XML tags
    for line in nlp_output.strip().splitlines():
        parts = line.split("\t")
        del parts[1]
        tag_attributes = {
            "text": parts[2].split("=")[1].strip('"'),
            "spans": "{}~{}".format(
                parts[3].split("=")[1].strip('"'), parts[4].split("=")[1].strip('"')
            ),
            "certainty": parts[5].split("=")[1].strip('"').lower(),
            "status": parts[6].split("=")[1].strip('"').lower(),
            "experiencer": parts[7].split("=")[1].strip('"').lower(),
            "id": parts[2].split("=")[1].strip('"')[:2].upper()
            + str(nlp_output.strip().splitlines().index(line)),
            # "CAM_criteria": "",
            "exclusion": "",
            "comment": "",
        }
        etree.SubElement(
            tags_element, parts[8].split("=")[1].strip('"').capitalize(), tag_attributes
        )
    rough_string = etree.tostring(root, pretty_print=True, encoding="UTF-8").decode(
        "utf-8"
    )
    return rough_string


def prepare_medtator_annotation(txt_dir, ann_dir):
    txt = glob.glob(txt_dir)
    nlp = glob.glob(ann_dir)
    for i in nlp:
        if is_file_size_greater_than_zero(i):
            raw_notes = read_txt(i.replace(".ann", "").replace("output", "input"))
            nlp_output = read_txt(i)
            combined_xml = create_combined_xml(raw_notes, nlp_output)
            outdir = i.replace(".ann", "").replace("output", "medtator") + ".xml"
            with open(outdir, "w") as text_file:
                text_file.write(combined_xml)


def main():
    # specify MedTagger output folder director
    raw_txt_dir = "/delirium/data/input"
    nlp_ann_dir = "/delirium/data/output"
    output_evidence(nlp_ann_dir + "/*.ann")
    prepare_medtator_annotation(raw_txt_dir + "/*.txt", nlp_ann_dir + "/*.ann")


if __name__ == "__main__":
    main()
